"""
Shared spatial regridding to the common East Africa 1° grid.

Two operators live here, used depending on the field:

* ``conservative_regrid`` — area-weighted (mass-conserving) remapping via xESMF.
  This is the operator used for **precipitation**, both for the observational
  references (CHIRPS/ERA5/TAMSAT) and for every model's daily rainfall, so models
  and truth are put on the common grid through the *same* mass-conserving
  operator. This matters when the native grid is finer than 1° (FourCastNet
  0.25°, ERA5/CHIRPS/TAMSAT) — bilinear point-sampling would bias totals and
  smear extremes — or coarser (NeuralGCM ~2.8°), where it distributes source-cell
  mass proportionally.
* ``to_ea_grid`` — bilinear interpolation, used for the *non-precip* instantaneous
  state fields saved when ``save_variables == "all"`` (temperature, winds, …),
  where a smooth interpolant is appropriate and mass conservation is meaningless.

Non-precip fields are treated as instantaneous snapshots (no temporal
accumulation) — the adapters select the valid lead-day time steps before calling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    level, …) are preserved. Source lat may be ascending or descending, and may
    use the ``latitude``/``longitude`` coordinate names (e.g. NeuralGCM).
    """
    # Normalise coordinate names → lat / lon
    ren = {k: v for k, v in (("longitude", "lon"), ("latitude", "lat"))
           if k in obj.dims or k in obj.coords}
    if ren:
        obj = obj.rename(ren)
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
    # The ``time`` coord may be a timedelta-from-init (GenCast/GraphCast/FCN) or
    # an absolute datetime (NeuralGCM); build the selector to match.
    time_is_delta = np.issubdtype(np.asarray(ds["time"].values).dtype, np.timedelta64)
    per_lead = []
    for lead in config.lead_days:
        if time_is_delta:
            sel = np.timedelta64(int(lead) * 24, "h")
        else:
            sel = np.datetime64(date) + np.timedelta64(int(lead) * 24, "h")
        snap = ds.sel(time=sel, method="nearest")
        per_lead.append(snap.drop_vars("time", errors="ignore"))

    stacked = xr.concat(per_lead, dim="lead_day").assign_coords(
        lead_day=list(config.lead_days)
    )
    stacked = to_ea_grid(stacked, config)
    stacked = stacked.expand_dims({"init_time": [np.datetime64(date, "ns")]})

    # Canonical dim order: init_time, sample, lead_day, [level/…], lat, lon.
    # Force lat/lon to be the trailing dims in that order — NeuralGCM emits
    # (lon, lat), so without this they would be transposed in the saved store.
    lead_order = [d for d in ("init_time", sample_dim, "lead_day") if d in stacked.dims]
    tail = [d for d in ("lat", "lon") if d in stacked.dims]
    mid  = [d for d in stacked.dims if d not in lead_order and d not in tail]
    return stacked.transpose(*lead_order, *mid, *tail)


# ── Conservative (mass-conserving) regridding ─────────────────────────────────
#
# Shared by the truth loaders (benchmark_ea/truth/*) and the model precip
# adapters (benchmark_ea/models/*), so precipitation from every source reaches
# the common 1° grid through one identical area-weighted operator. Weights depend
# only on the source/target grids and are cached to disk (keyed by a hash of both
# grids), so they are computed once per grid pair and reused across dates.


def _bounds_1d(coord: np.ndarray) -> np.ndarray:
    """Cell-edge bounds for a 1-D coordinate (midpoints, extrapolated at ends)."""
    coord = np.asarray(coord, dtype=np.float64)
    if coord.size < 2:
        raise ValueError("conservative regridding requires at least two grid points per axis")
    bounds = np.empty(coord.size + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    bounds[0]    = coord[0]  - 0.5 * (coord[1]  - coord[0])
    bounds[-1]   = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return bounds


def _rectilinear_grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return xr.Dataset(
        coords={
            "lat":   ("lat",   lat),
            "lon":   ("lon",   lon),
            "lat_b": ("lat_b", _bounds_1d(lat)),
            "lon_b": ("lon_b", _bounds_1d(lon)),
        }
    )


def _grid_hash(source_grid: xr.Dataset, target_grid: xr.Dataset) -> str:
    h = hashlib.blake2b(digest_size=10)
    for grid in (source_grid, target_grid):
        for name in ("lat", "lon", "lat_b", "lon_b"):
            h.update(np.ascontiguousarray(grid[name].values).view(np.uint8))
    return h.hexdigest()


def conservative_regrid(
    obj: xr.Dataset | xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    cache_dir,
    *,
    tag: str = "grid",
    subset_buffer: float | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Area-weighted (conservative) regrid of ``obj`` onto ``target_lat`` × ``target_lon``.

    Coordinate names are normalised to ``lat``/``lon`` and the latitude axis is
    flipped to ascending if needed, so sources using ``latitude``/``longitude`` or
    a descending grid (NeuralGCM, CHIRPS, …) are handled transparently. All
    non-spatial dims (sample, lead_day, time, …) are preserved/broadcast.

    Parameters
    ----------
    target_lat, target_lon : 1-D arrays of the destination grid cell centres.
    cache_dir : directory in which the xESMF weights file is cached.
    tag : short prefix for the weights filename (e.g. the model/obs name).
    subset_buffer : if given, first subset the source to the target bounding box
        expanded by this many degrees. Bounds memory when the source is global;
        the buffer must exceed the source cell size so every target cell stays
        fully covered. Leave ``None`` for already-subset sources (truth loaders).
    """
    try:
        import xesmf as xe
    except ImportError as exc:
        raise ImportError(
            "conservative regridding requires xesmf. Install benchmark-ea with "
            "its runtime dependencies, or install xesmf."
        ) from exc

    # Normalise coordinate names → lat / lon, ascending latitude.
    ren = {k: v for k, v in (("longitude", "lon"), ("latitude", "lat"))
           if k in obj.dims or k in obj.coords}
    if ren:
        obj = obj.rename(ren)
    obj = _ensure_ascending_lat(obj)
    obj = obj.transpose(..., "lat", "lon")

    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)

    if subset_buffer is not None:
        obj = obj.sel(
            lat=slice(float(target_lat.min()) - subset_buffer,
                      float(target_lat.max()) + subset_buffer),
            lon=slice(float(target_lon.min()) - subset_buffer,
                      float(target_lon.max()) + subset_buffer),
        )

    source_grid = _rectilinear_grid(obj.lat.values, obj.lon.values)
    target_grid = _rectilinear_grid(target_lat, target_lon)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / f"{tag}_conservative_{_grid_hash(source_grid, target_grid)}.nc"
    regridder = xe.Regridder(
        source_grid,
        target_grid,
        "conservative",
        filename=str(weights_path),
        reuse_weights=weights_path.exists(),
        unmapped_to_nan=True,
    )
    return regridder(obj, skipna=True)
