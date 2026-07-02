"""
NeuralGCM adapter — hybrid physics-ML model (Kochkov et al., Nature 2024).

Checkpoint : gs://neuralgcm/models/v1_precip/  (public, anonymous GCS)
ERA5 init  : ARCO-ERA5 public zarr on GCS (anonymous), same source as the
             GraphCast / GenCast adapters.
Engine     : JAX, via the `neuralgcm` package (`pip install neuralgcm`).

Which checkpoint
----------------
The base `v1/` models do NOT output precipitation. Only the `v1_precip/`
checkpoints do, and they are stochastic-only at 2.8°:
  * stochastic_precip_2_8_deg.pkl  → predicts precipitation directly   ← we use this
  * stochastic_evap_2_8_deg.pkl    → predicts precipitation − evaporation

Verified by the smoke test (neuralgcm 1.2.2)
--------------------------------------------
* Inputs the model needs: ``model.input_variables`` (geopotential, specific
  humidity, temperature, u/v wind, cloud ice/liquid water) + ``forcing_variables``
  (SST, sea-ice). Pressure levels: ``model.data_coords.vertical.centers`` (37).
* Native grid: 128×64 Gaussian (~2.8°, ~280 km) — coarse over East Africa; we
  regrid up to the common 1° grid like every other model.
* Precipitation output variable is **``precipitation_cumulative_mean``** = precip
  depth ACCUMULATED from init to each output day (metres). Per-day total = the
  consecutive difference ×1000 (mm/day); validated against CHIRPS in ``_to_canonical``.
* Forcings (SST / sea-ice) over the forecast horizon are prescribed from ERA5
  ("perfect forcing"), as in the public demo.
* Ensemble = the stochastic model run with different JAX RNG keys (per member).

Running it
----------
``neuralgcm`` lives in **user-site** with **JAX 0.6.2** here, while GraphCast/
GenCast use the conda env's JAX 0.4.30. So run this adapter **without**
``PYTHONNOUSERSITE=1`` (the GraphCast/GenCast wrappers set it, which would hide
user-site and fall back to the incompatible 0.4.30). A dedicated env is cleaner
long-term. Heavy imports are kept inside ``run_inference`` so this module imports
fine (e.g. for ``benchmark_ea.run``) even when ``neuralgcm`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea import regrid
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.models.base import ModelAdapter

# Official public checkpoints
_GCS_PRECIP_CKPT = "gs://neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl"


class NeuralGCMAdapter(ModelAdapter):
    """Stochastic NeuralGCM precipitation ensemble, regridded to the EA grid."""

    name        = "neuralgcm"
    is_ensemble = True

    def __init__(
        self,
        checkpoint: str = _GCS_PRECIP_CKPT,
        n_members:  int = 10,
        rng_seed:   int = 0,
    ):
        self.checkpoint = checkpoint
        self.n_members  = n_members
        self.rng_seed   = rng_seed

    # ───────────────────────────────────────────────────────────────────────
    def run_inference(self, config: BenchmarkConfig) -> Path:
        try:
            import gcsfs
            import jax
            import neuralgcm
            from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
        except ImportError as exc:  # keep the package optional
            raise ImportError(
                "neuralgcm (and dinosaur/jax) must be installed to run NeuralGCM.\n"
                "Install in a DEDICATED env (do not mix with the GraphCast/GenCast "
                "env):  pip install neuralgcm"
            ) from exc

        n_members = config.n_members or self.n_members

        # ── Load model from public GCS ────────────────────────────────────────
        print(f"Loading NeuralGCM checkpoint {self.checkpoint} …")
        import pickle
        fs = gcsfs.GCSFileSystem(token="anon")
        with fs.open(self.checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

        # ── ARCO-ERA5 (same source as the other adapters) ─────────────────────
        print("Connecting to ARCO-ERA5 …")
        arco = _connect_arco()

        out_dir  = self.predictions_path(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_lead = max(config.lead_days)
        dates    = pd.date_range(config.eval_start, config.eval_end, freq="D")

        for date in dates:
            zarr_path = out_dir / f"pred_{date.strftime('%Y-%m-%d')}.zarr"
            if not config.overwrite and self.should_skip(zarr_path, config.lead_days):
                print(f"  {date.date()} — skipping (exists)")
                continue

            print(f"  {date.date()} — preparing ERA5 initial conditions …")
            era5_on_grid = _era5_to_model_grid(
                arco, date, max_lead, model,
                horizontal_interpolation, spherical_harmonic, xarray_utils,
            )

            print(f"  {date.date()} — running {n_members}-member NeuralGCM ensemble …")
            members = []
            for m in range(n_members):
                pred_ds = _run_member(
                    model, era5_on_grid, date, max_lead,
                    rng_key=jax.random.PRNGKey(self.rng_seed + m),
                    jax=jax,
                )
                members.append(pred_ds)
            # stack members → (sample, time, lat, lon)
            predictions = xr.concat(members, dim="sample").assign_coords(
                sample=np.arange(n_members)
            )

            ds = ModelAdapter.assemble_output(
                _to_canonical(predictions, date, config),
                predictions if config.save_variables == "all" else None,
                date, config,
                precip_raw_vars=(_PRECIP_VAR,),
                sample_dim="sample",
            )
            ds.to_zarr(str(zarr_path), mode="w")
            print(f"  {date.date()} — saved → {zarr_path.name} ({list(ds.data_vars)})")

        return out_dir


# ── ARCO ERA5 ───────────────────────────────────────────────────────────────

def _connect_arco() -> xr.Dataset:
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    arco = xr.open_zarr(
        fs.get_mapper("gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"),
        consolidated=True, chunks={},
    )
    return arco


# ── ERA5 → model grid ─────────────────────────────────────────────────────────
# NeuralGCM's encode() expects ERA5 fields on the model's (2.8°) grid. The public
# demo: select the model's required variables/levels at the init time(s),
# conservatively regrid 0.25° → model grid, and fill NaNs.  This is the part most
# likely to need adjusting once we run it once and inspect `model.input_variables`.

def _era5_to_model_grid(arco, date, max_lead, model,
                        horizontal_interpolation, spherical_harmonic, xarray_utils):
    # Time stamps the model needs: the init time plus the forecast horizon (for
    # prescribed SST / sea-ice forcings). 6-hourly is the demo cadence.
    times  = pd.date_range(date, date + pd.Timedelta(days=max_lead), freq="6h")
    # Verified API: required fields are model.input_variables + forcing_variables;
    # pressure levels live in data_coords.vertical.centers (no model.pressure_levels).
    levels = [int(x) for x in np.asarray(model.data_coords.vertical.centers)]
    want   = list(model.input_variables) + list(model.forcing_variables)
    era5   = arco[want].sel(time=times, level=levels).compute()

    # Build the ERA5 (0.25°) source grid and conservatively regrid to the model
    # (~2.8° Gaussian) grid, then fill NaNs (ocean) by nearest neighbour.
    source = spherical_harmonic.Grid(
        latitude_nodes=era5.sizes["latitude"],
        longitude_nodes=era5.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(source, model.data_coords.horizontal)
    return xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(era5, regridder))


def _run_member(model, era5_on_grid, date, max_lead, rng_key, jax):
    """One stochastic NeuralGCM trajectory → xr.Dataset(time, lat, lon) of precip."""
    inputs   = model.inputs_from_xarray(era5_on_grid.isel(time=0))
    forcings = model.forcings_from_xarray(era5_on_grid.isel(time=0))
    state    = model.encode(inputs, forcings, rng_key)

    all_forcings = model.forcings_from_xarray(era5_on_grid)
    # Save once per day out to max_lead.
    _, predictions = model.unroll(
        state, all_forcings,
        steps=max_lead,
        timedelta=np.timedelta64(24, "h"),
        start_with_input=False,
    )
    times = pd.date_range(date + pd.Timedelta(days=1), date + pd.Timedelta(days=max_lead), freq="D")
    return model.data_to_xarray(predictions, times=times)


# ── Canonical conversion (precip → EA 1° grid, mm/day) ─────────────────────────
# Verified by smoke test + a raw-value probe vs CHIRPS: the v1_precip model emits
# `precipitation_cumulative_mean` (dims time, surface, lon, lat) = the precipitation
# depth ACCUMULATED from the init time to each output day, in metres. The raw EA-mean
# series grows monotonically (e.g. 0.0023, 0.0067, 0.0117 m over days 1-3), so the
# per-day total is the consecutive difference ×1000 → mm/day, which matches CHIRPS
# (~2.3 / 4.4 / 5.1 mm/day for 2024-03-01 +1/2/3 d).
_PRECIP_VAR = "precipitation_cumulative_mean"


def _to_canonical(predictions: xr.Dataset, date: pd.Timestamp,
                  config: BenchmarkConfig) -> xr.Dataset:
    """
    NeuralGCM stochastic precip → canonical (init_time, sample, lead_day, lat, lon).

    `predictions[_PRECIP_VAR]` is the cumulative-mean precip rate (m/day) with dims
    (sample, time, [surface,] lat/lon), where time = daily valid dates init+1…init+L.
    We de-accumulate to per-day mm/day, then subset to East Africa and interpolate
    to the 1° grid.
    """
    if _PRECIP_VAR not in predictions:
        raise KeyError(
            f"Expected precip variable {_PRECIP_VAR!r} not in NeuralGCM output: "
            f"{list(predictions.data_vars)}."
        )
    prec = predictions[_PRECIP_VAR]
    if "surface" in prec.dims:
        prec = prec.squeeze("surface", drop=True)
    rename = {}
    if "longitude" in prec.dims: rename["longitude"] = "lon"
    if "latitude"  in prec.dims: rename["latitude"]  = "lat"
    if rename:
        prec = prec.rename(rename)
    if float(prec.lat[0]) > float(prec.lat[-1]):
        prec = prec.isel(lat=slice(None, None, -1))

    # `precipitation_cumulative_mean` is the precipitation depth ACCUMULATED from
    # init to each output day (metres), so the per-day total is the consecutive
    # difference (the first day is the value itself). ×1000 → mm/day.
    # Verified against CHIRPS: diff×1000 ≈ 2.3/4.4/5.1 mm/day for 2024-03-01 +1/2/3d.
    daily_m  = xr.concat([prec.isel(time=0), prec.diff("time")], dim="time")
    daily_mm = (daily_m * 1000.0).clip(min=0).assign_coords(time=prec.time)

    lead_arrays = []
    for lead in config.lead_days:
        lead_arrays.append(daily_mm.isel(time=lead - 1))   # native (sample, lon, lat)

    # Native ~2.8° daily totals → EA 1° grid via the shared mass-conserving
    # operator (same as the observations). conservative_regrid normalises the
    # (lon, lat) orientation and the coarse coordinate names; a wide subset
    # buffer keeps every target cell covered by the coarse source cells.
    native = xr.concat(lead_arrays, dim="lead_day").assign_coords(
        lead_day=list(config.lead_days)
    )
    ea = regrid.conservative_regrid(
        native, config.lat_vals, config.lon_vals, config.regrid_weights_dir,
        tag="neuralgcm", subset_buffer=6.0,
    ).transpose(..., "lead_day", "lat", "lon")

    stacked = ea.values         # (sample, n_lead, lat, lon)
    return xr.Dataset({
        "total_precipitation": xr.DataArray(
            stacked[np.newaxis],
            dims=["init_time", "sample", "lead_day", "lat", "lon"],
            coords={
                "init_time": [np.datetime64(date, "ns")],
                "sample":    np.arange(stacked.shape[0]),
                "lead_day":  config.lead_days,
                "lat":       config.lat_vals,
                "lon":       config.lon_vals,
            },
            attrs={"units": "mm/day", "model": "neuralgcm_stochastic_precip_2.8deg"},
        )
    })
