"""
TAMSAT v3.1 daily rainfall estimates over Africa.

Source:  https://gws-access.jasmin.ac.uk/public/tamsat/tamsat3/data/daily/v3.1/
Format:  daily NetCDF, variable "rfe" in mm/day, 0.0375° grid over Africa.
Cover:   ~-38.5° to 38.5°N, ~-20° to 52°E.

Files are downloaded one per day and cached locally; the grid is conservatively
regridded to the target 1° model grid with xESMF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm


_BASE_URL = (
    "https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/v3.1/daily/"
)


def _daily_url(date: pd.Timestamp) -> str:
    y, m, d = date.year, date.month, date.day
    return f"{_BASE_URL}{y}/{m:02d}/rfe{y}_{m:02d}_{d:02d}.v3.1.nc"


def _daily_filename(date: pd.Timestamp) -> str:
    return f"tamsat_rfe_{date.strftime('%Y-%m-%d')}.nc"


def download_day(date: pd.Timestamp, cache_dir: Path) -> Path:
    """Download a single TAMSAT daily file; skip if already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / _daily_filename(date)
    if dest.exists():
        return dest
    url = _daily_url(date)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=date.strftime("%Y-%m-%d"), leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    return dest


def load(
    start: str,
    end: str,
    lat: np.ndarray,
    lon: np.ndarray,
    cache_dir: Union[str, Path],
    *,
    download_missing: bool = True,
) -> xr.DataArray:
    """
    Load TAMSAT v3.1 daily rainfall estimates, subset to the EA domain, and
    regrid to the target 1° grid.

    Parameters
    ----------
    start, end        : ISO date strings, inclusive ("2024-03-01", "2024-06-07").
    lat, lon          : 1-D ascending float arrays — the target model grid.
    cache_dir         : directory to store daily .nc files and regrid weights.
    download_missing  : if True, download any missing daily files automatically.

    Returns
    -------
    xr.DataArray  (time, lat, lon)  mm/day
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for a pre-regridded combined cache
    combined_cache = cache_dir / _combined_cache_name(start, end, lat, lon)
    if combined_cache.exists():
        da = xr.open_dataarray(combined_cache)
        da.attrs.update({"source": "TAMSAT v3.1", "units": "mm/day", "url": _BASE_URL})
        return da

    dates = pd.date_range(start, end, freq="D")
    slices: list[xr.DataArray] = []

    print(f"Loading TAMSAT {start} → {end} ({len(dates)} days) …")
    for date in tqdm(dates, desc="TAMSAT days"):
        path = cache_dir / _daily_filename(date)
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"TAMSAT {date.date()} not in cache ({path}). "
                    "Pass download_missing=True or run tamsat.download_day() first."
                )
            download_day(date, cache_dir)
        da_day = _open_and_subset(path, lat, lon)
        slices.append(da_day)

    combined = xr.concat(slices, dim="time")

    print("  Regridding TAMSAT 0.0375° → 1° …")
    regridded = _conservative_regrid(combined, lat, lon, cache_dir).astype(np.float32)
    regridded.attrs.update({"source": "TAMSAT v3.1", "units": "mm/day", "url": _BASE_URL})

    regridded.to_netcdf(combined_cache)
    print(f"  Cached combined TAMSAT → {combined_cache.name}")
    return regridded


def _open_and_subset(path: Path, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Open one TAMSAT daily file, normalise coords, spatial-subset."""
    ds = xr.open_dataset(path, engine="netcdf4")
    da = ds["rfe"]

    rename = {}
    if "latitude"  in da.dims: rename["latitude"]  = "lat"
    if "longitude" in da.dims: rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    # Ensure lat is ascending
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))

    buf = 1.0
    da = da.sel(
        lat=slice(float(lat[0]) - buf, float(lat[-1]) + buf),
        lon=slice(float(lon[0]) - buf, float(lon[-1]) + buf),
    )
    return da


def _conservative_regrid(
    da: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    cache_dir: Path,
) -> xr.DataArray:
    try:
        import xesmf as xe
    except ImportError as exc:
        raise ImportError(
            "TAMSAT conservative regridding requires xesmf. "
            "Install benchmark-ea with its runtime dependencies, or install xesmf."
        ) from exc

    target_lat = lat.astype(np.float64)
    target_lon = lon.astype(np.float64)
    source_grid = _rectilinear_grid(
        da.lat.values.astype(np.float64),
        da.lon.values.astype(np.float64),
    )
    target_grid = _rectilinear_grid(target_lat, target_lon)

    weights_path = cache_dir / f"tamsat_conservative_{_grid_hash(source_grid, target_grid)}.nc"
    regridder = xe.Regridder(
        source_grid,
        target_grid,
        "conservative",
        filename=str(weights_path),
        reuse_weights=weights_path.exists(),
        unmapped_to_nan=True,
    )
    return regridder(da, skipna=True)


def _rectilinear_grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "lat":   ("lat",   lat),
            "lon":   ("lon",   lon),
            "lat_b": ("lat_b", _bounds_1d(lat)),
            "lon_b": ("lon_b", _bounds_1d(lon)),
        }
    )


def _bounds_1d(coord: np.ndarray) -> np.ndarray:
    coord = coord.astype(np.float64)
    bounds = np.empty(coord.size + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    bounds[0]    = coord[0]  - 0.5 * (coord[1]  - coord[0])
    bounds[-1]   = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return bounds


def _grid_hash(source_grid: xr.Dataset, target_grid: xr.Dataset) -> str:
    h = hashlib.blake2b(digest_size=10)
    for grid in (source_grid, target_grid):
        for name in ("lat", "lon", "lat_b", "lon_b"):
            h.update(np.ascontiguousarray(grid[name].values).view(np.uint8))
    return h.hexdigest()


def _combined_cache_name(start: str, end: str, lat: np.ndarray, lon: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(start.encode())
    h.update(end.encode())
    h.update(lat.astype(np.float32).tobytes())
    h.update(lon.astype(np.float32).tobytes())
    return f"tamsat_rfe_{start}_{end}_{h.hexdigest()}.nc"
