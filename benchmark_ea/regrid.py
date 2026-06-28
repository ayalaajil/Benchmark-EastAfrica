"""
Shared spatial regridding to the common East Africa 1° grid.

Used both for the canonical precipitation field and, when
``save_variables == "all"``, for every other model output variable so the saved
zarr is small and consistent with the verification grid. Non-precip fields are
treated as instantaneous snapshots (no temporal accumulation) — the adapters are
responsible for selecting the valid lead-day time steps before calling here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from benchmark_ea.config import BenchmarkConfig


def _ensure_ascending_lat(obj: xr.Dataset | xr.DataArray):
    """Flip the lat axis to ascending if the source grid is descending."""
    if "lat" not in obj.coords or obj.sizes.get("lat", 0) < 2:
        return obj
    if float(obj.lat[0]) > float(obj.lat[-1]):
        return obj.isel(lat=slice(None, None, -1))
    return obj


def to_ea_grid(
    obj: xr.Dataset | xr.DataArray,
    config: BenchmarkConfig,
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """
    Subset to the East Africa bounding box and bilinearly interpolate onto the
    canonical 1° grid (``config.lat_vals`` × ``config.lon_vals``).

    Works on a DataArray or a whole Dataset; all non-spatial dims (sample, time,
    level, …) are preserved. Source lat may be ascending or descending.
    """
    obj = _ensure_ascending_lat(obj)
    obj = obj.sel(
        lat=slice(config.lat_min - 0.5, config.lat_max + 0.5),
        lon=slice(config.lon_min - 0.5, config.lon_max + 0.5),
    )
    return obj.interp(
        lat=config.lat_vals.astype(np.float64),
        lon=config.lon_vals.astype(np.float64),
        method=method,
    )


def extra_vars_to_canonical(
    raw: xr.Dataset,
    date,
    config: BenchmarkConfig,
    drop_vars: tuple[str, ...] = (),
    sample_dim: str | None = None,
) -> xr.Dataset:
    """
    Build the canonical ``(init_time, sample, lead_day, [level], lat, lon)``
    layout for every *non-precip* model variable, regridded to the EA grid.

    The input ``raw`` must carry a ``time`` coordinate expressed as a
    timedelta-from-init (``timedelta64``); the instantaneous state valid at lead
    day ``L`` is selected at ``L`` days. Any ``batch`` dim of length 1 is dropped.

    Parameters
    ----------
    drop_vars : variables to exclude (e.g. the precip fields handled separately).
    sample_dim : name of the ensemble member dim in ``raw`` (e.g. "sample"), or
        None for deterministic models. When ``config.extra_var_members == 'mean'``
        and this dim is present, the ensemble mean is stored (sample length 1).
    """
    keep = [v for v in raw.data_vars if v not in drop_vars]
    ds = raw[keep]

    if "batch" in ds.dims and ds.sizes["batch"] == 1:
        ds = ds.squeeze("batch", drop=True)

    # Collapse ensemble members for the extra vars if requested.
    if sample_dim is not None and sample_dim in ds.dims:
        if config.extra_var_members == "mean":
            ds = ds.mean(dim=sample_dim, keep_attrs=True)
        ds = ds if sample_dim in ds.dims else ds.expand_dims({sample_dim: [0]})
    else:
        sample_dim = "sample"
        ds = ds.expand_dims({sample_dim: [0]})

    # Instantaneous snapshot valid at each lead day, stacked on lead_day.
    per_lead = []
    for lead in config.lead_days:
        snap = ds.sel(time=np.timedelta64(int(lead) * 24, "h"), method="nearest")
        per_lead.append(snap.drop_vars("time", errors="ignore"))

    stacked = xr.concat(per_lead, dim="lead_day").assign_coords(
        lead_day=list(config.lead_days)
    )
    stacked = to_ea_grid(stacked, config)
    stacked = stacked.expand_dims({"init_time": [np.datetime64(date, "ns")]})

    # Canonical leading dim order: init_time, sample, lead_day, …
    lead_order = [d for d in ("init_time", sample_dim, "lead_day") if d in stacked.dims]
    rest = [d for d in stacked.dims if d not in lead_order]
    return stacked.transpose(*lead_order, *rest)
