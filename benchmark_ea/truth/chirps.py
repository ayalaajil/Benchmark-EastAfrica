"""
CHIRPS v2.0 daily precipitation — downloader, loader, and regridder.

Source:  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/
Format:  annual NetCDF, variable "precip" in mm/day, 0.05° global grid.
Fill:    -9999.0 (masked on load).

Notes on regridding
-------------------
CHIRPS is 0.05° (~5 km). At the 1° target resolution each output cell
averages ~20x20 source pixels. xESMF conservative regridding computes
area-weighted averages instead of sampling a single interpolation point.
"""

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm


_BASE_URL   = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
_FILL_VALUE = -9999.0


def _annual_filename(year: int) -> str:
    return f"chirps-v2.0.{year}.days_p05.nc"


def download(year: int, cache_dir: Union[str, Path]) -> Path:
    """
    Download the annual CHIRPS file for *year* to *cache_dir* if not present.

    Annual files are ~500-700 MB; they are never re-downloaded if the local
    copy already exists (no checksum, so delete the file to force a refresh).

    Returns the local path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / _annual_filename(year)
    if dest.exists():
        return dest

    url = _BASE_URL + _annual_filename(year)
    print(f"Downloading CHIRPS {year}  →  {dest.name}  ({url})")

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=str(year)
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    return dest


def load(
    start: str,
    end:   str,
    lat:   np.ndarray,
    lon:   np.ndarray,
    cache_dir: Union[str, Path],
    *,
    download_missing: bool = True,
) -> xr.DataArray:
    """
    Load CHIRPS daily precipitation, subset to the EA domain, and regrid to
    the target 1° grid.

    Parameters
    ----------
    start, end        : ISO date strings, inclusive ("2024-03-01", "2024-04-30").
    lat, lon          : 1-D ascending float arrays — the target model grid.
    cache_dir         : directory containing (or to receive) the annual .nc files.
    download_missing  : if True, fetch any missing annual files automatically.

    Returns
    -------
    xr.DataArray  (time, lat, lon)  mm/day
        NaN where CHIRPS has no data (ocean, fill-masked).
    """
    cache_dir = Path(cache_dir)
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    years = range(t0.year, t1.year + 1)

    slices: list[xr.DataArray] = []
    for year in years:
        path = cache_dir / _annual_filename(year)
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"CHIRPS {year} not in cache ({path}). "
                    "Pass download_missing=True or run chirps.download() first."
                )
            download(year, cache_dir)

        da = _open_and_subset(path, lat, lon)
        slices.append(da)

    combined = xr.concat(slices, dim="time")
    combined = combined.sel(time=slice(start, end))

    regridded = _conservative_regrid(combined, lat, lon, cache_dir).astype(np.float32)
    regridded.attrs.update({"source": "CHIRPS v2.0", "units": "mm/day", "url": _BASE_URL})
    return regridded


def _open_and_subset(path: Path, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Open one annual CHIRPS file, normalise coords, mask fill, spatial-subset."""
    ds = xr.open_dataset(path, engine="netcdf4")
    da = ds["precip"]

    # Mask fill value before any computation
    da = da.where(da != _FILL_VALUE)

    # Standardise coordinate names → lat / lon
    rename = {}
    if "latitude"  in da.dims: rename["latitude"]  = "lat"
    if "longitude" in da.dims: rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    # CHIRPS latitude runs 50 → -50 (descending); flip to ascending
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))

    # Spatial subset with 1° buffer to avoid interpolation edge effects
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
            "CHIRPS conservative regridding requires xesmf. "
            "Install benchmark-ea with its runtime dependencies, or install xesmf."
        ) from exc

    target_lat = lat.astype(np.float64)
    target_lon = lon.astype(np.float64)
    source_grid = _rectilinear_grid(
        da.lat.values.astype(np.float64),
        da.lon.values.astype(np.float64),
    )
    target_grid = _rectilinear_grid(target_lat, target_lon)

    weights_path = cache_dir / f"chirps_conservative_{_grid_hash(source_grid, target_grid)}.nc"
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
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "lat_b": ("lat_b", _bounds_1d(lat)),
            "lon_b": ("lon_b", _bounds_1d(lon)),
        }
    )


def _bounds_1d(coord: np.ndarray) -> np.ndarray:
    coord = coord.astype(np.float64)
    if coord.size < 2:
        raise ValueError("conservative regridding requires at least two grid points per axis")

    bounds = np.empty(coord.size + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    bounds[0] = coord[0] - 0.5 * (coord[1] - coord[0])
    bounds[-1] = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return bounds


def _grid_hash(source_grid: xr.Dataset, target_grid: xr.Dataset) -> str:
    h = hashlib.blake2b(digest_size=10)
    for grid in (source_grid, target_grid):
        for name in ("lat", "lon", "lat_b", "lon_b"):
            h.update(np.ascontiguousarray(grid[name].values).view(np.uint8))
    return h.hexdigest()
