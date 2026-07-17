"""
FourCastNet v2 (SFNO) deterministic adapter via NVIDIA earth2mip.

Precipitation pipeline
----------------------
FourCastNet v2 predicts atmospheric state (73 channels) but has NO native
precipitation output.  Precipitation is produced by the separate
PrecipitationAFNO diagnostic model (from the original FourCastNet paper) that
takes 20 of FCNv2's output channels as input and returns 6h-accumulated tp (m).

    FCNv2 IC ─► [FCNv2 73-ch state] ─► [PrecipAFNO] ─► tp  (per 6h step)

Weights (downloaded once on first use, no API key required):
    from earth2mip import registry
    from earth2mip.diagnostic.precipitation_afno import PrecipitationAFNO
    registry.get_model("e2mip://fcnv2_sm")
    PrecipitationAFNO.load_package()   # downloads precipitation_afno weights

Initial conditions
------------------
Loaded from ARCO-ERA5 (anonymous GCS) — same source as GenCast / GraphCast.
Relative humidity channels ("r{level}") are derived on-the-fly from
specific humidity and temperature using the Tetens formula.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea import regrid
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.models.base import ModelAdapter

_FCN_MODEL_URI  = "e2mip://fcnv2_sm"   # earth2mip registry name for FourCastNet v2 small
_FCN_STEP_HOURS = 6     # SFNO predicts in 6 h steps

# ── Channel → ARCO mapping ────────────────────────────────────────────────────

# Surface (height-based) channels.
# earth2mip uses "u10m" / "v10m" (with trailing 'm'), NOT "u10" / "v10".
_SURFACE_CHANNELS: dict[str, str] = {
    "u10m":  "10m_u_component_of_wind",
    "v10m":  "10m_v_component_of_wind",
    "u100m": "100m_u_component_of_wind",
    "v100m": "100m_v_component_of_wind",
    "t2m":   "2m_temperature",
    "sp":    "surface_pressure",
    "msl":   "mean_sea_level_pressure",
    "tcwv":  "total_column_water_vapour",
}

# Pressure-level channel prefix → ARCO variable name.
# "r" (relative humidity) is absent from ARCO; computed from q + T.
_PLEVEL_PREFIX: dict[str, str | None] = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "r": None,   # computed
    "q": "specific_humidity",
    "w": "vertical_velocity",
}

_PLEVEL_RE = re.compile(r"^([a-z]+)(\d+)$")


def _parse_fcn_channel(ch: str) -> tuple[str | None, int | None, bool]:
    """
    Parse an earth2mip FCN channel name to (arco_var, level_hpa, needs_rh).

    Surface names (e.g. "u10m", "t2m") are looked up directly.
    Pressure-level names (e.g. "u850", "r500") are split into prefix + level.
    Raises ValueError for unrecognised channels.
    """
    if ch in _SURFACE_CHANNELS:
        return _SURFACE_CHANNELS[ch], None, False

    m = _PLEVEL_RE.match(ch)
    if m:
        prefix, level_str = m.group(1), m.group(2)
        level = int(level_str)
        if prefix == "r":
            return None, level, True
        arco_var = _PLEVEL_PREFIX.get(prefix)
        if arco_var is not None:
            return arco_var, level, False

    raise ValueError(
        f"Cannot map FCN channel '{ch}' to an ARCO variable.\n"
        f"  Known surface channels: {sorted(_SURFACE_CHANNELS)}\n"
        f"  Known pressure-level prefixes: {sorted(_PLEVEL_PREFIX)}\n"
        f"  Add an entry in fourcastnet.py to resolve this."
    )


# ── Adapter ───────────────────────────────────────────────────────────────────

class FourCastNetAdapter(ModelAdapter):
    """Deterministic 6h-step precipitation from FourCastNet v2 + PrecipAFNO. sample=1."""

    name        = "fourcastnet"
    is_ensemble = False

    def __init__(self, model_uri: str = _FCN_MODEL_URI):
        self.model_uri = model_uri

    def run_inference(self, config: BenchmarkConfig) -> Path:
        try:
            from earth2mip import registry
            from earth2mip.networks import fcnv2_sm
            from earth2mip.diagnostic.precipitation_afno import PrecipitationAFNO
        except ImportError as exc:
            raise ImportError(
                "earth2mip is not installed in the active Python environment.\n"
                "Use run_inference.sh which activates the aim-graphcast conda env."
            ) from exc

        print(f"Loading FourCastNet v2 ({self.model_uri}) …")
        fcn_pkg = registry.get_model(self.model_uri)

        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # FCNv2 weights.tar contains ruamel.yaml types which are rejected.
        # Patch torch.load to use weights_only=False for this trusted checkpoint.
        import torch
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            fcn = fcnv2_sm.load(fcn_pkg, device="cuda")
        finally:
            torch.load = _orig_load  # restore original after loading

        print("Loading PrecipitationAFNO diagnostic …")
        precip_pkg   = PrecipitationAFNO.load_package()
        precip_model = PrecipitationAFNO.load_diagnostic(precip_pkg, device="cuda:0")

        # Precompute indices of PrecipAFNO's 20 input channels in FCNv2's output
        fcn_out_names    = list(fcn.out_channel_names)
        precip_in_idx    = [fcn_out_names.index(ch)
                            for ch in precip_model.in_channel_names]

        print("Connecting to ARCO-ERA5 …")
        arco = _connect_arco()

        # Validate channel mapping once before looping
        print("Validating ARCO channel mapping …")
        for ch in fcn.in_channel_names:
            _parse_fcn_channel(ch)

        out_dir  = self.predictions_path(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_lead = max(config.lead_days)
        dates    = pd.date_range(config.eval_start, config.eval_end, freq="D")

        for date in dates:
            zarr_path = out_dir / f"pred_{date.strftime('%Y-%m-%d')}.zarr"
            if not config.overwrite and self.should_skip(zarr_path, config.lead_days):
                print(f"  {date.date()} — skipping (exists, all lead days present)")
                continue

            print(f"  {date.date()} — loading ARCO initial conditions …")
            ic = _arco_to_fcn_ic(arco, date, fcn.in_channel_names, device="cuda")

            print(f"  {date.date()} — running FCNv2 + PrecipAFNO …")
            precip_preds, state_preds = _run_fcn(
                fcn, precip_model, precip_in_idx, ic,
                date, max_lead,
                config=config, save_state=config.save_variables == "all",
            )

            canonical = _to_canonical(precip_preds, date, config)
            ds = ModelAdapter.assemble_output(
                canonical, state_preds, date, config,
                precip_raw_vars=(),     # state_preds carries no precip field
                sample_dim=None,
            )
            ds.to_zarr(str(zarr_path), mode="w")
            print(f"  {date.date()} — saved → {zarr_path.name} ({list(ds.data_vars)})")

        return out_dir


# ── ARCO initial-condition loader ─────────────────────────────────────────────

def _connect_arco() -> xr.Dataset:
    import gcsfs
    fs   = gcsfs.GCSFileSystem(token="anon")
    arco = xr.open_zarr(
        fs.get_mapper(
            "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        ),
        consolidated=True, chunks={},
    )
    print(f"  ARCO range: {str(arco.time.values[0])[:10]} → "
          f"{str(arco.time.values[-1])[:10]}")
    return arco


def _arco_to_fcn_ic(
    arco:          xr.Dataset,
    date:          pd.Timestamp,
    channel_names: list[str],
    device:        str = "cuda",
) -> "torch.Tensor":
    """
    Build FCNv2 initial-condition tensor (1, 1, n_channels, 721, 1440).

    ARCO has 721 lat points (90 → -90, includes south pole), which matches
    the FCNv2 grid exactly (equiangular_lat_lon_grid(721, 1440)).
    No interpolation or row-dropping needed.

    The TimeLoop Inference class expects (B, n_history_levels, C, lat, lon).
    FCNv2 has n_history=0, so n_history_levels=1 → shape (1, 1, 73, 721, 1440).
    """
    import torch

    arrays = []
    for ch in channel_names:
        arco_var, level, is_rh = _parse_fcn_channel(ch)

        if is_rh:
            arr = _relative_humidity(arco, date, level)
        elif level is not None:
            arr = (
                arco[arco_var]
                .sel(time=date, level=level)
                .values
                .astype(np.float32)
            )
        else:
            arr = (
                arco[arco_var]
                .sel(time=date)
                .values
                .astype(np.float32)
            )

        if arr.ndim != 2:
            arr = arr.squeeze()
        arrays.append(arr)   # (721, 1440) — full ARCO lat, matches FCNv2 exactly

    stacked = np.stack(arrays, axis=0).astype(np.float32)  # (C, 721, 1440)
    # TimeLoop expects (B, n_history_levels, C, lat, lon)
    return torch.from_numpy(stacked[np.newaxis, np.newaxis]).to(device)


def _relative_humidity(arco: xr.Dataset, date: pd.Timestamp, level: int) -> np.ndarray:
    """Compute relative humidity (%) from ARCO specific humidity + temperature."""
    q = arco["specific_humidity"].sel(time=date, level=level).values.astype(np.float64)
    T = arco["temperature"].sel(time=date, level=level).values.astype(np.float64)

    # Saturation vapor pressure (hPa) via Tetens / Murray (1967)
    T_c = T - 273.15
    es  = 6.1078 * np.exp(17.269 * T_c / (T_c + 237.29))

    # Actual vapor pressure from specific humidity
    e  = q * level / (0.622 + q * 0.378)

    return np.clip(100.0 * e / es, 0.0, 100.0).astype(np.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_fcn(
    fcn,
    precip_model,
    precip_in_idx:  list[int],
    ic:             "torch.Tensor",
    date:           pd.Timestamp,
    max_lead:       int,
    config:         "BenchmarkConfig | None" = None,
    save_state:     bool = False,
) -> "tuple[xr.Dataset, xr.Dataset | None]":
    """
    Autoregressive FCNv2 rollout with PrecipAFNO applied at each 6h step.

    ic       : (1, 1, 73, 721, 1440) initial condition on CUDA
    Returns  : (precip_ds, state_ds)
      precip_ds : total_precipitation_6hr (n_steps, 720, 1440) in m, absolute time
      state_ds  : every FCNv2 output channel as a variable, subset to a box
                  around East Africa, with a timedelta-from-init ``time`` axis —
                  or None when ``save_state`` is False.

    Grid note: FCNv2 outputs 721 lat points (90 → -90).  PrecipAFNO needs 720
    (90 → -89.75).  Slice [:, :, :720, :] to drop the south-pole row.

    Yield order: Inference yields (IC at t=0), then t+6h, t+12h, …
    The first yield is the IC itself — skip it, collect from yield 1 onwards.
    """
    import torch

    n_steps     = max_lead * (24 // _FCN_STEP_HOURS)
    init_time   = date.to_pydatetime().replace(tzinfo=datetime.timezone.utc)

    step_arrays = []
    step_times  = []

    # State grid: FCNv2 720-row grid (90 → -89.75) after dropping the pole row.
    import earth2mip.grid as egrid
    pg = egrid.equiangular_lat_lon_grid(720, 1440, includes_south_pole=False)
    lat = np.array(pg.lat, dtype=np.float32)
    lon = np.array(pg.lon, dtype=np.float32)

    # Precompute an East-Africa lat/lon index window so we only keep a small
    # box of the global state in memory (lat is descending: 90 → -89.75).
    state_names, state_steps = list(fcn.out_channel_names), []
    if save_state and config is not None:
        lat_keep = np.where((lat >= config.lat_min - 2.0) & (lat <= config.lat_max + 2.0))[0]
        lon_keep = np.where((lon >= config.lon_min - 2.0) & (lon <= config.lon_max + 2.0))[0]
        lat_sl = slice(int(lat_keep[0]), int(lat_keep[-1]) + 1)
        lon_sl = slice(int(lon_keep[0]), int(lon_keep[-1]) + 1)
        state_lat, state_lon = lat[lat_sl], lon[lon_sl]

    with torch.no_grad():
        for step, (time, output, _) in enumerate(fcn(init_time, ic)):
            if step == 0:
                continue   # first yield is IC at t=0, not a forecast step

            # output: (1, 73, 721, 1440) — slice :720 to match PrecipAFNO grid
            precip_in = output[:, precip_in_idx, :720, :]   # (1, 20, 720, 1440)
            tp = precip_model(precip_in)                     # (1, 1, 720, 1440) in m
            step_arrays.append(tp[0, 0].cpu().numpy())
            step_times.append(time)

            if save_state and config is not None:
                # (73, lat_box, lon_box) — subset on GPU before moving to CPU.
                box = output[0, :, :720, :][:, lat_sl, lon_sl].cpu().numpy()
                state_steps.append(box)

            if len(step_arrays) >= n_steps:
                break

    precip_ds = xr.Dataset({
        f"total_precipitation_{_FCN_STEP_HOURS}hr": xr.DataArray(
            np.stack(step_arrays, axis=0),   # (n_steps, 720, 1440) in metres
            dims=["time", "lat", "lon"],
            coords={
                "time": [np.datetime64(t.replace(tzinfo=None), "ns")
                         for t in step_times],
                "lat":   lat,
                "lon":   lon,
            },
        )
    })

    state_ds = None
    if save_state and config is not None and state_steps:
        t0 = np.datetime64(date, "ns")
        rel = np.array(
            [np.datetime64(t.replace(tzinfo=None), "ns") - t0 for t in step_times],
            dtype="timedelta64[ns]",
        )
        state = np.stack(state_steps, axis=0)   # (n_steps, 73, lat_box, lon_box)
        state_ds = xr.Dataset(
            {
                name: xr.DataArray(
                    state[:, ci], dims=["time", "lat", "lon"],
                    coords={"time": rel, "lat": state_lat, "lon": state_lon},
                )
                for ci, name in enumerate(state_names)
            }
        )

    return precip_ds, state_ds


# ── Post-processing ────────────────────────────────────────────────────────────

def _to_canonical(
    predictions: xr.Dataset,
    date:        pd.Timestamp,
    config:      BenchmarkConfig,
) -> xr.Dataset:
    """
    Accumulate 6h FourCastNet precipitation to daily totals (mm/day).

    FCNv2 lat is descending (90 → -89.75) — flipped to ascending before interp.
    """
    precip_var    = f"total_precipitation_{_FCN_STEP_HOURS}hr"
    prec          = predictions[precip_var]    # (n_steps, lat, lon) in m
    steps_per_day = 24 // _FCN_STEP_HOURS       # 4

    lead_arrays = []
    for lead in config.lead_days:
        daily_m = sum(
            prec.isel(time=(lead - 1) * steps_per_day + s).values
            for s in range(steps_per_day)
        )   # (lat, lon) in metres
        daily_m = np.clip(daily_m, 0.0, None)

        step_da = xr.DataArray(
            daily_m * 1000.0,   # m → mm/day
            coords={"lat": prec.lat, "lon": prec.lon},
            dims=["lat", "lon"],
        )
        lead_arrays.append(step_da)   # native 0.25° (lat, lon)

    # Native 0.25° daily totals → EA 1° grid via the shared mass-conserving
    # operator (same as the observations). Ascending-lat and bbox subsetting are
    # handled inside conservative_regrid; done once for all leads.
    native = xr.concat(lead_arrays, dim="lead_day").assign_coords(
        lead_day=list(config.lead_days)
    )
    ea = regrid.conservative_regrid(
        native, config.lat_vals, config.lon_vals, config.regrid_weights_dir,
        tag="fourcastnet_v2", subset_buffer=4.0,
    ).transpose("lead_day", "lat", "lon")

    stacked = ea.values[np.newaxis, np.newaxis]  # (1,1,n_lead,lat,lon)

    return xr.Dataset({
        "total_precipitation": xr.DataArray(
            stacked,
            dims=["init_time", "sample", "lead_day", "lat", "lon"],
            coords={
                "init_time": [np.datetime64(date, "ns")],
                "sample":    [0],
                "lead_day":  config.lead_days,
                "lat":       config.lat_vals,
                "lon":       config.lon_vals,
            },
            attrs={"units": "mm/day", "model": "fourcastnet_v2"},
        )
    })
