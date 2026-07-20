"""
Shared data loaders for the East Africa skill analyses — single source of truth.

Loads model predictions, a truth/climatology reference on the model grid, and
aligns forecast/observation pairs by valid date. ``run_verification.py``,
``ea_common.py`` and the standalone analysis scripts all import from here so that
every entry point loads and aligns data identically.

Array conventions:
  preds[model] : DataArray(init_time, sample, lead_day, lat, lon)  — mm/day
  obs_2d       : {date: (lat, lon) array}  daily rainfall, NaN over ocean
"""

from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import tamsat as tamsat_io

# name → (loader module, cache sub-directory) for load_truth
_TRUTH_IO = {
    "chirps": (chirps_io, "chirps"),
    "tamsat": (tamsat_io, "tamsat"),
}

_PRED_FILENAME_RE = re.compile(r"pred_(\d{4}-\d{2}-\d{2})\.zarr$")


def _init_date_from_path(path) -> pd.Timestamp:
    """The init date for a ``pred_YYYY-MM-DD.zarr`` file, parsed from its
    filename — the authoritative source of truth for init_time throughout
    this codebase (it's how these files are matched and organized), used in
    preference to the store's own init_time coordinate.

    NeuralGCM's writer has been observed to save a corrupted, undecodable
    init_time value (the same garbage int64 in every single file,
    independent of date) while every other coordinate/variable in the same
    store is intact; deriving init_time from the filename sidesteps that
    corruption instead of trying to repair it in place.
    """
    m = _PRED_FILENAME_RE.search(str(path))
    if not m:
        raise ValueError(f"cannot parse init date from filename: {path}")
    return pd.Timestamp(m.group(1))


def open_pred_da(path, var="total_precipitation"):
    """Open one ``pred_YYYY-MM-DD.zarr`` and return its ``var`` DataArray with
    init_time set from the filename (see ``_init_date_from_path``) — opening
    with ``decode_times=False`` so a corrupted init_time encoding in one
    model's files can't crash loading of any model's data."""
    ds = xr.open_zarr(path, decode_times=False)
    da = ds[var]
    return da.assign_coords(init_time=[_init_date_from_path(path)])


def load_predictions(pred_dir, models):
    """{model: DataArray(init_time, sample, lead_day, lat, lon)} of precipitation.

    Concatenates every ``pred_2024-*.zarr`` under ``<pred_dir>/<model>/`` on
    init_time. Guards against mixing zarrs on different grids (e.g. stale global
    files alongside freshly-regenerated East Africa ones), which would otherwise
    silently misalign on concat.
    """
    print("Loading model predictions …")
    preds = {}
    for m in models:
        files = sorted(glob.glob(f"{pred_dir}/{m}/pred_2024-*.zarr"))
        if not files:
            raise FileNotFoundError(f"No prediction files found for {m} in {pred_dir}/{m}/")
        parts = [open_pred_da(f) for f in files]
        grids = {(p.sizes["lat"], p.sizes["lon"]) for p in parts}
        if len(grids) > 1:
            raise ValueError(
                f"{m}: prediction files have inconsistent lat/lon grids {grids} in "
                f"{pred_dir}/{m}/ — clear stale files and regenerate."
            )
        members = {p.sizes.get("sample", 1) for p in parts}
        if len(members) > 1:
            raise ValueError(
                f"{m}: prediction files have inconsistent ensemble sizes {members} in "
                f"{pred_dir}/{m}/ — concat would pad missing members with NaN and "
                f"silently poison every ensemble score. Clear stale files and regenerate."
            )
        preds[m] = xr.concat(parts, dim="init_time")
        print(f"  {m:15s}  {dict(zip(preds[m].dims, preds[m].shape))}")
    return preds


def load_truth(name, start, obs_end, lat, lon, data_dir, download_missing=False):
    """Load a truth/climatology source on the model grid.

    Parameters
    ----------
    name : "chirps" or "tamsat".
    start, obs_end : first / last date to load (obs must cover init END + max lead).
    lat, lon : target 1-D grid arrays.
    data_dir : root data dir; the source is cached under ``<data_dir>/<name>``.

    Returns
    -------
    obs_2d : {date: (lat, lon) array}  daily rainfall, NaN over ocean
    lat, lon : 1-D coordinate arrays of the loaded field
    """
    name = name.lower()
    if name not in _TRUTH_IO:
        raise ValueError(f"truth must be one of {list(_TRUTH_IO)}, got {name!r}")
    io, subdir = _TRUTH_IO[name]
    da = io.load(start, obs_end, lat, lon, f"{data_dir}/{subdir}",
                 download_missing=download_missing)
    obs_2d = {pd.Timestamp(t).date(): da.sel(time=t).values for t in da.time.values}
    return obs_2d, da.lat.values, da.lon.values


def gather_pairs(preds, model, obs_2d, init_dates, lead_day=1):
    """Align a model's forecasts with truth at a given lead day, keeping the grid.

    Per-cell (spatial) pairing that keeps the lat/lon grid and does no masking;
    ocean cells stay NaN in ``ob_all`` so they drop out of area averages.

    Returns
    -------
    fc_all : (case, sample, lat, lon)
    ob_all : (case, lat, lon)
    or ``(None, None)`` when no case has matching observations.
    """
    fc_da = preds[model].sel(lead_day=lead_day)
    fc_list, ob_list = [], []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values   # (sample, lat, lon)
        except Exception:
            continue
        fc_list.append(fc)
        ob_list.append(obs_2d[vd])
    if not fc_list:
        return None, None
    return np.stack(fc_list), np.stack(ob_list)
