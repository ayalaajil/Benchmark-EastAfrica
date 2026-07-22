"""
CHIRPS v2.0 daily precipitation — downloader, loader, and regridder.

Source:  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/

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

from benchmark_ea import regrid


_BASE_URL   = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
_FILL_VALUE = -9999.0


def _annual_filename(year: int) -> str:
    return f"chirps-v2.0.{year}.days_p05.nc"


def download(year: int, cache_dir: Union[str, Path]) -> Path:

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / _annual_filename(year)
    
    if dest.exists():
        return dest

    url = _BASE_URL + _annual_filename(year)
    print(f"Downloading CHIRPS {year}  →  {dest.name}  ({url})")

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=str(year)
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        if total and tmp.stat().st_size != total:
            raise IOError(
                f"CHIRPS {year}: incomplete download "
                f"({tmp.stat().st_size} of {total} bytes)"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

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


def _target_grid_hash(lat: np.ndarray, lon: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(np.asarray(lat, dtype=np.float64).tobytes())
    h.update(np.asarray(lon, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def load_year_ea(
    year: int,
    lat:  np.ndarray,
    lon:  np.ndarray,
    cache_dir: Union[str, Path],
    *,
    download_missing: bool = True,
) -> xr.DataArray:
    """
    Load ONE year of CHIRPS already regridded to the EA target grid, caching only
    the tiny regridded result (~1 MB) and DISCARDING the ~1 GB global annual file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ea_cache = cache_dir / f"chirps_ea1deg_{year}_{_target_grid_hash(lat, lon)}.nc"
    if ea_cache.exists():
        da = xr.open_dataarray(ea_cache)
        da.load()                      
        return da

    glob_path = cache_dir / _annual_filename(year)
    if not glob_path.exists():
        if not download_missing:
            raise FileNotFoundError(
                f"CHIRPS {year} not cached (no EA cache, no global file). "
                "Pass download_missing=True."
            )
        download(year, cache_dir)

    da = _open_and_subset(glob_path, lat, lon)
    da = _conservative_regrid(da, lat, lon, cache_dir).astype(np.float32)
    da = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    da.attrs.update({"source": "CHIRPS v2.0", "units": "mm/day", "url": _BASE_URL})

    da.to_netcdf(ea_cache)             
    glob_path.unlink(missing_ok=True) 
    return da


def _open_and_subset(path: Path, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Open one annual CHIRPS file, normalise coords, mask fill, spatial-subset."""
    ds = xr.open_dataset(path, engine="netcdf4")
    da = ds["precip"]

    da = da.where(da != _FILL_VALUE)

    rename = {}
    if "latitude"  in da.dims: rename["latitude"]  = "lat"
    if "longitude" in da.dims: rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

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
    """Area-weighted regrid to the model grid via the shared benchmark operator."""
    return regrid.conservative_regrid(da, lat, lon, cache_dir, tag="chirps")
