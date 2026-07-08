"""
GenCast ensemble adapter.

ERA5 init  : ARCO-ERA5 public zarr on GCS (anonymous)

Inference
--------------
1. Load GenCast Mini checkpoint + norm stats + static fields from GCS.
2. Build haiku-transformed, JIT-compiled forward function.
3. Connect to ARCO-ERA5.
4. For each init_time in the eval period:
   a. Load ERA5 at t0 and t0+12h (two input states).
   b. Build multi-day batch: inputs + n_lead*2 target slots at 12h steps.
   c. Run chunked_prediction_generator_multiple_runs for n_members samples.
   d. Sum pairs of 12h predictions to daily totals (mm/day).
   e. Subset to EA domain, write canonical zarr.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea import regrid
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.models.base import ModelAdapter

# ── GCS paths ─────────────────────────────────────────────────────────────────

_GCS_BUCKET = "dm_graphcast"
_GCS_PREFIX = "gencast/"

# ── ERA5 variable groups (matches gencast.TASK) ───────────────────────────────

_PLEVEL_VARS = [
    "temperature", "geopotential",
    "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
]
_SURF_VARS = [
    "2m_temperature", "sea_surface_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
_STEP_HOURS = 12


# ── Adapter ───────────────────────────────────────────────────────────────────

class GenCastAdapter(ModelAdapter):
    """
    Ensemble precipitation forecasts from GenCast Mini (default) or the
    full operational checkpoint.
    """

    name        = "gencast"
    is_ensemble = True

    def __init__(
        self,
        n_members:   int        = 10,
        rng_seed:    int        = 0,
        params_file: str | None = None,   # None → auto-select Mini
    ):
        self.n_members   = n_members
        self.rng_seed    = rng_seed
        self.params_file = params_file

    def run_inference(self, config: BenchmarkConfig) -> Path:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        from graphcast import (
            casting, checkpoint, data_utils, gencast,
            nan_cleaning, normalization, rollout,
        )

        # ── Load model ────────────────────────────────────────────────────────
        print("Loading GenCast model from GCS …")
        params, state, ckpt, stats, static_vars = _load_model(self.params_file)
        task_config = ckpt.task_config

        # ── Build JIT forward fn ──────────────────────────────────────────────
        run_forward_jit = _build_predictor_fn(params, state, ckpt, stats)

        # ── ARCO connection ───────────────────────────────────────────────────
        print("Connecting to ARCO-ERA5 …")
        arco   = _connect_arco()
        levels = list(task_config.pressure_levels)

        out_dir  = self.predictions_path(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_lead = max(config.lead_days)
        dates    = pd.date_range(config.eval_start, config.eval_end, freq="D")

        for date in dates:
            zarr_path = out_dir / f"pred_{date.strftime('%Y-%m-%d')}.zarr"
            if not config.overwrite and self.should_skip(zarr_path, config.lead_days):
                print(f"  {date.date()} — skipping (exists, all lead days present)")
                continue

            print(f"  {date.date()} — building {max_lead}-day batch …")
            batch = _build_batch(date, arco, static_vars, levels, max_lead)

            inputs, targets, forcings = _extract(batch, task_config, max_lead)

            n_members = config.n_members
            print(f"  {date.date()} — running {n_members}-member ensemble …")
            rngs = np.stack(
                [jax.random.fold_in(jax.random.PRNGKey(self.rng_seed), i)
                 for i in range(n_members)],
                axis=0,
            )

            save_all = config.save_variables == "all"
            # When saving every variable we retain the full per-member state, but
            # subset it to a box around East Africa *before* moving off the GPU so
            # memory stays bounded (the native global field is discarded). For the
            # default precip-only path we keep only total_precipitation_12hr, so
            # the GPU buffer for each chunk is freed step-by-step.
            ea_box = dict(
                lat=slice(config.lat_min - 2.0, config.lat_max + 2.0),
                lon=slice(config.lon_min - 2.0, config.lon_max + 2.0),
            )
            by_sample: dict[int, list] = {}
            for chunk in rollout.chunked_prediction_generator_multiple_runs(
                predictor_fn=run_forward_jit,
                rngs=rngs,
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
                num_samples=n_members,
                num_steps_per_chunk=1,
            ):
                s = int(chunk.coords["sample"].values)
                if save_all:
                    sub = chunk.drop_vars("sample", errors="ignore").sel(**ea_box)
                    # Rebuild as a plain numpy-backed Dataset; np.asarray triggers
                    # jax.device_get so the GPU buffer can be freed.
                    sub = xr.Dataset(
                        {v: (sub[v].dims, np.asarray(sub[v].data))
                         for v in sub.data_vars},
                        coords={c: np.asarray(sub[c].data) for c in sub.coords},
                    )
                    by_sample.setdefault(s, []).append(sub)
                else:
                    tp = chunk["total_precipitation_12hr"]
                    coords = {k: v for k, v in tp.coords.items() if k != "sample"}
                    by_sample.setdefault(s, []).append(
                        xr.Dataset({"total_precipitation_12hr":
                                    xr.DataArray(np.asarray(tp), dims=tp.dims, coords=coords)})
                    )

            trajectories = [xr.concat(by_sample[s], dim="time")
                            for s in sorted(by_sample)]
            predictions = xr.concat(trajectories, dim="sample")
            predictions = predictions.assign_coords(sample=np.arange(len(trajectories)))

            ds = ModelAdapter.assemble_output(
                _to_canonical(predictions, date, config),
                predictions if save_all else None,
                date, config,
                precip_raw_vars=("total_precipitation_12hr",),
                sample_dim="sample",
            )
            ds.to_zarr(str(zarr_path), mode="w")
            print(f"  {date.date()} — saved → {zarr_path.name} ({list(ds.data_vars)})")

        return out_dir


# ── GCS / model loading ────────────────────────────────────────────────────────

def _gcs_client():
    from google.cloud import storage
    return storage.Client.create_anonymous_client()


def _load_model(params_file: str | None):
    """Download checkpoint, norm stats, and static fields from GCS."""
    from graphcast import checkpoint, gencast
    import jax

    client = _gcs_client()
    bucket = client.get_bucket(_GCS_BUCKET)
    prefix = f"{_GCS_PREFIX}params/"

    # ── checkpoint ────────────────────────────────────────────────────────────
    if params_file is None:
        blobs = [b for b in bucket.list_blobs(prefix=prefix)
                 if b.name != prefix]
        names = [b.name.removeprefix(prefix) for b in blobs]
        mini  = [n for n in names if "Mini" in n]
        params_file = mini[0] if mini else names[0]
        print(f"  Auto-selected checkpoint: {params_file}")

    blob = bucket.blob(f"{prefix}{params_file}")
    print(f"  Downloading {params_file} …")
    with blob.open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    params = ckpt.params
    state  = {}

    # Fix attention type for non-TPU backends
    if jax.default_backend() != "tpu":
        st_cfg = ckpt.denoiser_architecture_config.sparse_transformer_config
        if st_cfg.attention_type == "splash_mha":
            st_cfg.attention_type = "triblockdiag_mha"
            st_cfg.mask_type      = "full"

    # ── norm stats ────────────────────────────────────────────────────────────
    print("  Downloading normalization stats …")
    stat_names = ["diffs_stddev_by_level", "mean_by_level",
                  "stddev_by_level", "min_by_level"]
    stats = {}
    for name in stat_names:
        with bucket.blob(f"{_GCS_PREFIX}stats/{name}.nc").open("rb") as f:
            stats[name] = xr.load_dataset(f, decode_timedelta=False).compute()

    # ── static fields ─────────────────────────────────────────────────────────
    print("  Loading static fields …")
    static_vars = _load_static_fields(bucket)

    return params, state, ckpt, stats, static_vars


def _load_static_fields(bucket) -> xr.Dataset:
    prefix = f"{_GCS_PREFIX}dataset/"
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name.removeprefix(prefix)
        if name and "res-1.0" in name and "era5" in name.lower():
            with blob.open("rb") as f:
                ds = xr.load_dataset(f, decode_timedelta=False).compute()
            return ds[["land_sea_mask", "geopotential_at_surface"]]

    raise FileNotFoundError(
        f"No ERA5 1° static fields in gs://{_GCS_BUCKET}/{prefix}. "
        "Expected a file with 'res-1.0' and 'era5' in the name."
    )


# ── Predictor function ────────────────────────────────────────────────────────

def _build_predictor_fn(params, state, ckpt, stats):
    """Build haiku-transformed, JIT-compiled GenCast predictor."""
    import haiku as hk
    import jax
    from graphcast import casting, gencast, nan_cleaning, normalization

    def _make_predictor():
        pred = gencast.GenCast(
            task_config=ckpt.task_config,
            denoiser_architecture_config=ckpt.denoiser_architecture_config,
            sampler_config=ckpt.sampler_config,
            noise_config=ckpt.noise_config,
            noise_encoder_config=ckpt.noise_encoder_config,
        )
        pred = normalization.InputsAndResiduals(
            pred,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
        )
        pred = nan_cleaning.NaNCleaner(
            predictor=pred,
            reintroduce_nans=True,
            fill_value=stats["min_by_level"],
            var_to_clean="sea_surface_temperature",
        )
        return pred

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return _make_predictor()(inputs, targets_template=targets_template,
                                 forcings=forcings)

    return jax.jit(
        lambda rng, inputs, targets_template, forcings:
            run_forward.apply(params, state, rng, inputs, targets_template, forcings)[0]
    )


# ── ARCO ERA5 ─────────────────────────────────────────────────────────────────

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


# ── Batch building ─────────────────────────────────────────────────────────────

_TARGET_LAT = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
_TARGET_LON = np.arange(0.0,  360.0, 1.0, dtype=np.float32)


def _build_batch(
    date:        pd.Timestamp,
    arco:        xr.Dataset,
    static_vars: xr.Dataset,
    levels:      list,
    max_lead:    int,
) -> xr.Dataset:
    """
    Build a complete GenCast input batch for *date*.

    Timestamps: t0, t0+12h  (two input states covering input_duration=24h)
                t0+24h, ..., t0+(max_lead*2)*12h  (target slots)
    All target variable values are NaN'd in the targets_template before inference.
    """
    from graphcast import data_utils

    n_target   = max_lead * 2
    timestamps = [date + pd.Timedelta(hours=i * _STEP_HOURS)
                  for i in range(n_target + 2)]
    t0_ns    = np.datetime64(date, "ns")
    rel_times = np.array(
        [(np.datetime64(t, "ns") - t0_ns) for t in timestamps],
        dtype="timedelta64[ns]",
    )

    def _to_1deg(ds: xr.Dataset) -> xr.Dataset:
        rename = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")]
                  if k in ds.coords}
        ds = ds.rename(rename)
        if float(ds.lat[0]) > float(ds.lat[-1]):
            ds = ds.isel(lat=slice(None, None, -1))
        ds = ds.interp(lat=_TARGET_LAT.astype(float),
                       lon=_TARGET_LON.astype(float), method="linear")
        return ds.assign_coords(time=rel_times)

    def _add_batch(ds: xr.Dataset) -> xr.Dataset:
        return ds.assign({v: ds[v].expand_dims({"batch": 1}, axis=0)
                          for v in ds.data_vars if "batch" not in ds[v].dims})

    print(f"    Loading ERA5 pressure-level vars …")
    plevel = _to_1deg(
        arco[_PLEVEL_VARS].sel(time=timestamps, level=levels).compute()
        .assign_coords(level=arco.level.sel(level=levels).astype("int32"))
    )

    print(f"    Loading ERA5 surface vars …")
    surf = _to_1deg(arco[_SURF_VARS].sel(time=timestamps).compute())

    print(f"    Loading 12h accumulated precipitation …")
    precip_chunks = []
    for t in timestamps:
        window = (
            arco["total_precipitation"]
            .sel(time=slice(t - pd.Timedelta("11h"), t))
            .sum("time", keep_attrs=True)
            .compute()
        )
        precip_chunks.append(window.expand_dims({"time": [t]}))
    precip = _to_1deg(
        xr.concat(precip_chunks, dim="time")
        .rename("total_precipitation_12hr")
        .to_dataset()
    )

    merged = xr.merge([
        _add_batch(plevel),
        _add_batch(surf),
        _add_batch(precip),
        static_vars,
    ])

    # Add datetime coord so data_utils.add_derived_vars can compute forcings
    merged = merged.assign_coords(
        datetime=xr.DataArray(
            np.array([[np.datetime64(t, "ns") for t in timestamps]]),
            dims=["batch", "time"],
        )
    )
    data_utils.add_derived_vars(merged)   # adds year/day progress sin/cos

    return merged


def _extract(batch, task_config, max_lead: int):
    """Split batch into inputs / targets_template / forcings."""
    from graphcast import data_utils
    n_steps = max_lead * 2
    return data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice(
            f"{_STEP_HOURS}h", f"{n_steps * _STEP_HOURS}h"
        ),
        **dataclasses.asdict(task_config),
    )


# ── Post-processing ────────────────────────────────────────────────────────────

def _to_canonical(
    predictions: xr.Dataset,
    date:        pd.Timestamp,
    config:      BenchmarkConfig,
) -> xr.Dataset:
    """
    Convert GenCast 12h-step predictions to canonical benchmark format.

    predictions: (batch=1, sample, time, lat, lon)  total_precipitation_12hr in m
    Returns:     (1, n_members, n_lead, lat, lon)    mm/day
    """
    prec = predictions["total_precipitation_12hr"].isel(batch=0, missing_dims="ignore")

    lead_arrays = []
    for lead in config.lead_days:
        t_am = pd.Timedelta(hours=(2 * lead - 1) * _STEP_HOURS)
        t_pm = pd.Timedelta(hours=2 * lead * _STEP_HOURS)
        daily_mm = (
            prec.sel(time=t_am, method="nearest") +
            prec.sel(time=t_pm, method="nearest")
        ) * 1000.0   # m → mm/day
        daily_mm = daily_mm.clip(min=0)
        lead_arrays.append(daily_mm)   # native (sample, lat, lon)

    # Native 1° daily totals → EA 1° grid via the shared mass-conserving operator
    # (same as the observations). Regrids all members and leads in one call.
    native = xr.concat(lead_arrays, dim="lead_day").assign_coords(
        lead_day=list(config.lead_days)
    )
    ea = regrid.conservative_regrid(
        native, config.lat_vals, config.lon_vals, config.regrid_weights_dir,
        tag="gencast", subset_buffer=4.0,
    ).transpose("sample", "lead_day", "lat", "lon")

    stacked = ea.values   # (n_members, n_lead, lat, lon)

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
            attrs={"units": "mm/day", "model": "gencast"},
        )
    })
