"""
Verification-specific data loaders.

Loads the optional climatology baseline predictions and the three observational
references (CHIRPS/ERA5/TAMSAT), and builds the per-date lookup dicts the
verification pipeline scores against. The generic prediction/truth loaders live
in benchmark_ea.analysis_io; these are the pieces specific to run_verification.
"""

import glob

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import era5 as era5_io
from benchmark_ea.truth import tamsat as tamsat_io


def load_climatology_reference(pred_dir):
    """
    Load the climatology baseline predictions if present, for CRPS skill scores.

    Returns the total_precipitation DataArray (init_time, sample, lead_day,
    lat, lon) or None when the climatology predictions have not been generated.
    """
    files = sorted(glob.glob(f"{pred_dir}/climatology/pred_2024-*.zarr"))
    if not files:
        return None
    parts = [xr.open_zarr(f)["total_precipitation"] for f in files]
    grids = {(p.sizes["lat"], p.sizes["lon"]) for p in parts}
    if len(grids) > 1:
        raise ValueError(
            f"climatology: inconsistent lat/lon grids {grids} in "
            f"{pred_dir}/climatology/ — clear stale files and regenerate."
        )
    members = {p.sizes.get("sample", 1) for p in parts}
    if len(members) > 1:
        raise ValueError(
            f"climatology: inconsistent ensemble sizes {members} in "
            f"{pred_dir}/climatology/ — concat would pad missing members with "
            f"NaN and silently poison CRPS. Clear stale files and regenerate."
        )
    return xr.concat(parts, dim="init_time")


def load_observations(config, obs_end, output_dir):
    print("\nLoading observations …")
    start = "2024-03-01"
    chirps_da = chirps_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                               config.chirps_cache_dir, download_missing=False)
    print(f"  CHIRPS  {dict(zip(chirps_da.dims, chirps_da.shape))}")

    era5_da = era5_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                           config.data_dir + "/era5", download_missing=True)
    print(f"  ERA5    {dict(zip(era5_da.dims, era5_da.shape))}")

    tamsat_da = tamsat_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                               config.data_dir + "/tamsat", download_missing=True)
    print(f"  TAMSAT  {dict(zip(tamsat_da.dims, tamsat_da.shape))}")

    return chirps_da, era5_da, tamsat_da


def build_lookup_dicts(chirps_da, era5_da, tamsat_da):
    chirps_2d = {pd.Timestamp(t).date(): chirps_da.sel(time=t).values
                 for t in chirps_da.time.values}
    era5_2d   = {pd.Timestamp(t).date(): era5_da.sel(time=t).values
                 for t in era5_da.time.values}
    tamsat_2d = {pd.Timestamp(t).date(): tamsat_da.sel(time=t).values
                 for t in tamsat_da.time.values}

    chirps_lookup  = {d: float(np.nanmean(v)) for d, v in chirps_2d.items()}
    era5_lookup    = {d: float(np.nanmean(v)) for d, v in era5_2d.items()}
    tamsat_lookup  = {d: float(np.nanmean(v)) for d, v in tamsat_2d.items()}

    return (chirps_2d, era5_2d, tamsat_2d,
            chirps_lookup, era5_lookup, tamsat_lookup)
