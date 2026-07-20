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

from benchmark_ea.analysis_io import open_pred_da
from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import era5 as era5_io
from benchmark_ea.truth import tamsat as tamsat_io
from benchmark_ea.verification.mask import apply_land_mask, land_mask


def load_climatology_reference(pred_dir):
    """
    Load the climatology baseline predictions if present, for CRPS skill scores.

    Returns the total_precipitation DataArray (init_time, sample, lead_day,
    lat, lon) or None when the climatology predictions have not been generated.
    """
    files = sorted(glob.glob(f"{pred_dir}/climatology/pred_2024-*.zarr"))
    if not files:
        return None
    parts = [open_pred_da(f) for f in files]
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


def load_observations(config, start, obs_end, output_dir):
    """Load CHIRPS/ERA5/TAMSAT for [start, obs_end]. ``start`` must match the
    verification init-date range (not just its MAM subset) — otherwise
    seasons outside MAM (e.g. JF) would have no observations at all to score
    against on a full-year run."""
    print("\nLoading observations …")
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
    """Build the per-date {date: (lat, lon)} lookups, all masked to the same
    common land cells.

    CHIRPS is NaN over ocean at the source; ERA5 (a reanalysis) is not, so
    without an explicit common mask ERA5-verified scores would silently pool
    ocean cells that every CHIRPS/TAMSAT-verified score excludes. The mask is
    derived from CHIRPS validity (see ``verification.mask``) and applied to
    all three references here, once, so every downstream gather sees an
    identical cell set regardless of which reference it scores against.
    """
    chirps_2d = {pd.Timestamp(t).date(): chirps_da.sel(time=t).values
                 for t in chirps_da.time.values}
    era5_2d   = {pd.Timestamp(t).date(): era5_da.sel(time=t).values
                 for t in era5_da.time.values}
    tamsat_2d = {pd.Timestamp(t).date(): tamsat_da.sel(time=t).values
                 for t in tamsat_da.time.values}

    mask = land_mask(chirps_da)
    chirps_2d = apply_land_mask(chirps_2d, mask)
    era5_2d   = apply_land_mask(era5_2d, mask)
    tamsat_2d = apply_land_mask(tamsat_2d, mask)

    chirps_lookup  = {d: float(np.nanmean(v)) for d, v in chirps_2d.items()}
    era5_lookup    = {d: float(np.nanmean(v)) for d, v in era5_2d.items()}
    tamsat_lookup  = {d: float(np.nanmean(v)) for d, v in tamsat_2d.items()}

    return (chirps_2d, era5_2d, tamsat_2d,
            chirps_lookup, era5_lookup, tamsat_lookup)
