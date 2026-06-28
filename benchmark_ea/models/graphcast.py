"""
GraphCast-small deterministic adapter.

All inference code lives here — no dependency on AIM-for-Scale.

Checkpoint : gs://dm_graphcast/  (public, anonymous GCS)
  Weights  : "GraphCast_small - ERA5 1979-2015 - resolution 1.0 -
               pressure levels 13 - mesh 4 - precipitation input and output.npz"
  Stats    : stats/{diffs_stddev,mean,stddev}_by_level.nc
  Statics  : dataset/source-era5_date-*_res-1.0_levels-13_*.nc
ERA5 init  : ARCO-ERA5 public zarr on GCS (anonymous)

Install requirements
--------------------
    pip install "graphcast @ git+https://github.com/google-deepmind/graphcast"
    pip install dm-haiku "jax[cuda12_pip]" gcsfs google-cloud-storage

Inference flow
--------------
1. Load GraphCast-small checkpoint + norm stats + static fields from GCS.
2. Build haiku-transformed, JIT-compiled forward function.
3. Connect to ARCO-ERA5.
4. For each init_time:
   a. Load ERA5 at t-step_h and t from ARCO.
   b. Build batch: ERA5 vars + forcings (TISR, year/day progress).
   c. Run chunked_prediction autoregressive rollout to max lead.
   d. Accumulate step_hours precipitation steps to daily totals.
   e. Write canonical zarr (sample=1, deterministic).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.models.base import ModelAdapter


# ── GCS paths ─────────────────────────────────────────────────────────────────

GCS_BUCKET = "dm_graphcast"

GCS_PARAMS_PREFIX = "params/"
GCS_SMALL_PARAMS = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - "
    "precipitation input and output.npz"
)
GCS_STATS_NAMES = (
    "stats/diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
)

# ERA5 variables required as inputs (TASK_13_PRECIP_OUT, minus forcings and statics)
_PLEVEL_VARS = (
    "temperature", "geopotential",
    "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
)
_SURF_VARS = (
    "2m_temperature", "mean_sea_level_pressure",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
)


# ── Adapter ───────────────────────────────────────────────────────────────────

class GraphCastAdapter(ModelAdapter):
    """
    Deterministic precipitation forecasts from GraphCast-small (1° / 13 levels).
    sample=1 in the output zarr.
    """

    name        = "graphcast"
    is_ensemble = False

    def __init__(self, params_file: str = GCS_SMALL_PARAMS):
        self.params_file = params_file

    # ── main entry point ──────────────────────────────────────────────────────

    def run_inference(self, config: BenchmarkConfig) -> Path:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        from graphcast import casting, checkpoint, graphcast as gc, normalization, rollout

        # ── Load model ────────────────────────────────────────────────────────
        print("Loading GraphCast-small from GCS …")
        params, state, model_config, task_config, diffs_std, mean, std, static_vars = (
            _load_model_and_stats(self.params_file)
        )

        # ── Build JIT-compiled forward fn ─────────────────────────────────────
        def _construct():
            pred = gc.GraphCast(model_config, task_config)
            pred = normalization.InputsAndResiduals(
                pred,
                diffs_stddev_by_level=diffs_std,
                mean_by_level=mean,
                stddev_by_level=std,
            )
            pred = casting.Bfloat16Cast(pred)
            return pred

        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            return _construct()(inputs, targets_template=targets_template, forcings=forcings)

        run_forward_jit = jax.jit(
            lambda rng, inputs, targets_template, forcings:
                run_forward.apply(params, state, rng, inputs, targets_template, forcings)[0]
        )

        # ── ARCO connection ───────────────────────────────────────────────────
        print("Connecting to ARCO-ERA5 …")
        arco = _connect_arco()

        out_dir  = self.predictions_path(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_lead = max(config.lead_days)
        dates    = pd.date_range(config.eval_start, config.eval_end, freq="D")

        # Autoregressive step size = precipitation accumulation window (both 6h).
        # input_duration="12h" means the two input frames span 12h (each 6h apart),
        # NOT that the step is 12h. See GCS example batch: time=[0,6,12] hours.
        precip_var, acc_hours = _precip_var_from_task(task_config)
        step_hours = acc_hours  # 6h for all GraphCast variants

        for date in dates:
            zarr_path = out_dir / f"pred_{date.strftime('%Y-%m-%d')}.zarr"
            if not config.overwrite and self.should_skip(zarr_path, config.lead_days):
                print(f"  {date.date()} — skipping (exists, all lead days present)")
                continue

            print(f"  {date.date()} — building batch (step={step_hours}h, precip={acc_hours}h) …")
            batch = _build_gc_batch(
                date, arco, static_vars, task_config, step_hours, max_lead,
                precip_var=precip_var, acc_hours=acc_hours,
            )

            inputs, targets_template, forcings = _extract(batch, task_config, step_hours, max_lead)

            print(f"  {date.date()} — running GraphCast rollout …")
            predictions = rollout.chunked_prediction(
                run_forward_jit,
                rng=jax.random.PRNGKey(0),
                inputs=inputs,
                targets_template=targets_template * np.nan,
                forcings=forcings,
            )

            canonical = _to_canonical(
                predictions, date, config, step_hours, precip_var, acc_hours,
            )
            ds = ModelAdapter.assemble_output(
                canonical, predictions, date, config,
                precip_raw_vars=(precip_var,),
                sample_dim=None,
            )
            ds.to_zarr(str(zarr_path), mode="w")
            print(f"  {date.date()} — saved → {zarr_path.name} ({list(ds.data_vars)})")

        return out_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _precip_var_from_task(task_config) -> tuple[str, int]:
    """
    Find the precipitation target variable and its accumulation window (hours).
    e.g. "total_precipitation_6hr" → ("total_precipitation_6hr", 6)
    """
    for v in task_config.target_variables:
        if v.startswith("total_precipitation_") and v.endswith("hr"):
            acc_hours = int(v.split("_")[-1].removesuffix("hr"))
            return v, acc_hours
    raise ValueError(
        f"No total_precipitation_*hr variable found in task target_variables: "
        f"{task_config.target_variables}"
    )


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_and_stats(params_file: str):
    """Download checkpoint + norm stats + static fields from GCS (anonymous)."""
    import gcsfs
    from graphcast import checkpoint, graphcast as gc

    fs = gcsfs.GCSFileSystem(token="anon")

    print(f"  Downloading {params_file} …")
    with fs.open(f"{GCS_BUCKET}/{GCS_PARAMS_PREFIX}{params_file}", "rb") as f:
        ckpt = checkpoint.load(f, gc.CheckPoint)

    params       = ckpt.params
    state        = {}
    model_config = ckpt.model_config
    task_config  = ckpt.task_config

    print("  Downloading normalization stats …")
    stats = {}
    for name in GCS_STATS_NAMES:
        with fs.open(f"{GCS_BUCKET}/{name}", "rb") as f:
            stats[name] = xr.load_dataset(f, decode_timedelta=False).compute()

    print("  Downloading static fields …")
    static_vars = _load_static_fields(fs)

    return (
        params, state, model_config, task_config,
        stats[GCS_STATS_NAMES[0]],   # diffs_stddev
        stats[GCS_STATS_NAMES[1]],   # mean
        stats[GCS_STATS_NAMES[2]],   # stddev
        static_vars,
    )


def _load_static_fields(fs) -> xr.Dataset:
    """Find and load ERA5 1° static fields (land_sea_mask, geopotential_at_surface)."""
    prefix = f"{GCS_BUCKET}/dataset/"
    for path in fs.ls(prefix):
        name = path.removeprefix(prefix)
        if "res-1.0" in name and "era5" in name.lower():
            print(f"  Loading static fields from {name} …")
            with fs.open(path, "rb") as f:
                ds = xr.load_dataset(f, decode_timedelta=False).compute()
            return ds[["land_sea_mask", "geopotential_at_surface"]]

    raise FileNotFoundError(
        f"No ERA5 1° static fields found in gs://{GCS_BUCKET}/dataset/. "
        "Expected a file with 'res-1.0' and 'era5' in the name."
    )


# ── ARCO ERA5 loading ─────────────────────────────────────────────────────────

def _connect_arco() -> xr.Dataset:
    import gcsfs
    fs   = gcsfs.GCSFileSystem(token="anon")
    arco = xr.open_zarr(
        fs.get_mapper("gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"),
        consolidated=True, chunks={},
    )
    print(f"  ARCO range: {str(arco.time.values[0])[:10]} → {str(arco.time.values[-1])[:10]}")
    return arco


def _build_gc_batch(
    date:       pd.Timestamp,
    arco:       xr.Dataset,
    static:     xr.Dataset,
    task_config,
    step_hours: int,
    max_lead:   int,
    *,
    precip_var: str,
    acc_hours:  int,
) -> xr.Dataset:
    """
    Build a complete dataset for GraphCast inference at *date*.

    timestamps: t-step_h  (second input), t (first/latest input),
                t+step_h, t+2*step_h, ..., t+max_lead*24h  (targets)
    """
    from graphcast import data_utils

    n_target   = max_lead * (24 // step_hours)
    timestamps = (
        [date - pd.Timedelta(hours=step_hours), date]
        + [date + pd.Timedelta(hours=(i + 1) * step_hours) for i in range(n_target)]
    )

    # ERA5 levels matching the task pressure levels
    levels = list(task_config.pressure_levels)
    rename = {"latitude": "lat", "longitude": "lon"}

    def _load_and_regrid(var_list, times):
        raw = arco[list(var_list)].sel(time=times).compute()
        raw = raw.rename({k: v for k, v in rename.items() if k in raw.coords})
        if float(raw.lat[0]) > float(raw.lat[-1]):
            raw = raw.isel(lat=slice(None, None, -1))
        return raw

    target_lat = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    target_lon = np.arange(0.0,  360.0, 1.0, dtype=np.float32)

    plevel = (
        arco[list(_PLEVEL_VARS)].sel(time=timestamps, level=levels)
        .compute()
        .interp(latitude=target_lat.astype(float),
                longitude=target_lon.astype(float), method="linear")
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    if float(plevel.lat[0]) > float(plevel.lat[-1]):
        plevel = plevel.isel(lat=slice(None, None, -1))

    surf = (
        arco[list(_SURF_VARS)].sel(time=timestamps).compute()
        .interp(latitude=target_lat.astype(float),
                longitude=target_lon.astype(float), method="linear")
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    if float(surf.lat[0]) > float(surf.lat[-1]):
        surf = surf.isel(lat=slice(None, None, -1))

    # Accumulated precipitation: use acc_hours from the task config
    # (e.g. acc_hours=6 for total_precipitation_6hr, regardless of step_hours).
    precip_da = _load_precip_accumulated(arco, timestamps, acc_hours, target_lat, target_lon)

    # Build absolute-time dataset, then add derived variables
    # (year/day progress, TOA solar radiation).
    merged = xr.merge([
        plevel.assign_coords(time=timestamps),
        surf.assign_coords(time=timestamps),
        xr.Dataset({precip_var: precip_da}),
        static,
    ])

    # Add batch dimension where needed
    for var in merged.data_vars:
        if "lat" in merged[var].dims and "batch" not in merged[var].dims:
            merged[var] = merged[var].expand_dims({"batch": 1}, axis=0)

    # Forcings: year/day progress + TISR — computed from absolute datetimes
    merged = merged.assign_coords(
        datetime=xr.DataArray(
            np.array([[np.datetime64(t, "ns") for t in timestamps]]),
            dims=["batch", "time"],
        )
    )
    data_utils.add_derived_vars(merged)
    data_utils.add_tisr_var(merged)

    # Replace absolute time axis with relative timedelta from t=date
    t0_ns  = np.datetime64(date, "ns")
    rel_ns = np.array(
        [(np.datetime64(t, "ns") - t0_ns) for t in timestamps],
        dtype="timedelta64[ns]",
    )
    merged = merged.assign_coords(time=rel_ns)
    return merged


def _load_precip_accumulated(
    arco:       xr.Dataset,
    timestamps: list,
    step_hours: int,
    lat:        np.ndarray,
    lon:        np.ndarray,
) -> xr.DataArray:
    """Load step_hours-accumulated precipitation from ARCO hourly values."""
    chunks = []
    for t in timestamps:
        window = (
            arco["total_precipitation"]
            .sel(time=slice(t - pd.Timedelta(hours=step_hours - 1), t))
            .sum("time", keep_attrs=True)
            .compute()
        )
        chunks.append(window.expand_dims({"time": [t]}))

    da = xr.concat(chunks, dim="time")
    da = (
        da.interp(latitude=lat.astype(float), longitude=lon.astype(float), method="linear")
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))
    return da


def _extract(batch, task_config, step_hours: int, max_lead: int):
    """Thin wrapper around data_utils.extract_inputs_targets_forcings."""
    from graphcast import data_utils
    n_steps = max_lead * (24 // step_hours)
    return data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice(
            f"{step_hours}h", f"{n_steps * step_hours}h"
        ),
        **dataclasses.asdict(task_config),
    )


# ── Post-processing ────────────────────────────────────────────────────────────

def _to_canonical(
    predictions: xr.Dataset,
    date:        pd.Timestamp,
    config:      BenchmarkConfig,
    step_hours:  int,
    precip_var:  str,
    acc_hours:   int,
) -> xr.Dataset:
    """
    Accumulate GraphCast precipitation predictions to daily totals.

    Each prediction step outputs acc_hours of accumulated precip (in metres).
    Daily total = sum of all steps whose output window falls within the lead day.
    With step_hours=12 and acc_hours=6 there are 2 steps per day, each giving
    6h of precip, so daily total = sum of 2 steps (12h captured out of 24h).
    """
    prec = predictions[precip_var].isel(batch=0, missing_dims="ignore")
    # prec: (time, lat, lon), time = relative timedeltas from init

    # Number of model steps per 24h calendar day
    steps_per_day = 24 // step_hours
    lead_arrays   = []

    for lead in config.lead_days:
        steps_mm = []
        for s in range(steps_per_day):
            offset = pd.Timedelta(hours=((lead - 1) * 24) + (s + 1) * step_hours)
            step   = prec.sel(time=offset, method="nearest")
            steps_mm.append(step.values * 1000.0)   # m → mm

        daily_mm = np.clip(np.sum(steps_mm, axis=0), 0, None)  # (lat, lon) mm/day

        # Subset to EA domain and regrid to exact config grid
        step_da = xr.DataArray(
            daily_mm,
            coords={"lat": prec.lat, "lon": prec.lon},
            dims=["lat", "lon"],
        )
        step_da = (
            step_da
            .sel(
                lat=slice(config.lat_min - 0.5, config.lat_max + 0.5),
                lon=slice(config.lon_min - 0.5, config.lon_max + 0.5),
            )
            .interp(
                lat=config.lat_vals.astype(np.float64),
                lon=config.lon_vals.astype(np.float64),
                method="linear",
            )
        )
        lead_arrays.append(step_da.values)   # (lat, lon)

    # GraphCast is deterministic → sample=1
    # Shape: (1, n_leads, lat, lon) → add sample + init_time dims
    stacked = np.stack(lead_arrays, axis=0)[np.newaxis, np.newaxis]  # (1, 1, n_lead, lat, lon)

    return xr.Dataset(
        {
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
                attrs={"units": "mm/day", "model": "graphcast_small"},
            )
        }
    )
