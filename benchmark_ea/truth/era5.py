"""
ERA5 daily total precipitation from ARCO-ERA5 (public GCS zarr).

Source : gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
Variable: ``total_precipitation``  [m per 1-h period]
Method  : sum 24 hourly values per UTC day → daily total; convert m → mm/day;
          conservative regrid 0.25° → 1° with xESMF (weights cached locally).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


_ARCO_URI = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def _connect_arco() -> xr.Dataset:
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    return xr.open_zarr(
        fs.get_mapper(_ARCO_URI),
        consolidated=True,
        chunks={},
    )


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
    Load ERA5 daily total precipitation for a date range, regridded to the 1° target grid.

    Parameters
    ----------
    start, end        : ISO date strings, inclusive ("2024-03-01", "2024-05-31").
    lat, lon          : 1-D ascending float arrays — the 1° target model grid.
    cache_dir         : directory to store the cached NetCDF.
    download_missing  : if True, fetch from ARCO-ERA5 when cache is absent.

    Returns
    -------
    xr.DataArray  (time, lat, lon)  mm/day
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / _cache_filename(start, end, lat, lon)
    if cache_path.exists():
        da = xr.open_dataarray(cache_path)
        da.attrs.update({"source": "ERA5 via ARCO-ERA5", "units": "mm/day"})
        return da

    if not download_missing:
        raise FileNotFoundError(
            f"ERA5 cache not found: {cache_path}. "
            "Pass download_missing=True to fetch from ARCO-ERA5."
        )

    da = _download_and_regrid(start, end, lat, lon, cache_dir)
    da.to_netcdf(cache_path)
    print(f"  Cached ERA5 → {cache_path.name}")
    return da


def _download_and_regrid(
    start: str,
    end: str,
    lat: np.ndarray,
    lon: np.ndarray,
    cache_dir: Path,
) -> xr.DataArray:
    print(f"Connecting to ARCO-ERA5 …")
    arco = _connect_arco()

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    # Load all hours in the date range (inclusive of end-day last hour)
    time_start = t0.strftime("%Y-%m-%dT00:00")
    time_end   = t1.strftime("%Y-%m-%dT23:00")

    buf = 1.0
    lat_sl = slice(float(lat[-1]) + buf, float(lat[0]) - buf)  # descending in ARCO
    lon_sl = slice(float(lon[0]) - buf, float(lon[-1]) + buf)

    print(f"  Downloading total_precipitation {start} → {end} (East Africa) …")
    tp_hourly = (
        arco["total_precipitation"]
        .sel(time=slice(time_start, time_end),
             latitude=lat_sl,
             longitude=lon_sl)
        .compute()
    )

    # Flip lat to ascending, rename to lat/lon
    if float(tp_hourly.latitude[0]) > float(tp_hourly.latitude[-1]):
        tp_hourly = tp_hourly.isel(latitude=slice(None, None, -1))
    tp_hourly = tp_hourly.rename({"latitude": "lat", "longitude": "lon"})

    # Sum 24 hourly values per UTC day → daily total in metres
    tp_daily = tp_hourly.resample(time="1D").sum("time")  # (days, lat, lon) in m

    # Convert m → mm
    tp_daily = tp_daily * 1000.0

    # Conservative regrid 0.25° → 1°
    print("  Regridding 0.25° → 1° …")
    tp_1deg = _conservative_regrid(tp_daily, lat, lon, cache_dir).astype(np.float32)
    tp_1deg.attrs.update({"source": "ERA5 via ARCO-ERA5", "units": "mm/day"})
    return tp_1deg


# ── Regridding helpers (mirrors chirps.py) ────────────────────────────────────

def _conservative_regrid(
    da: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    cache_dir: Path,
) -> xr.DataArray:
    try:
        import xesmf as xe
    except ImportError as exc:
        raise ImportError("ERA5 regridding requires xesmf.") from exc

    target_lat = lat.astype(np.float64)
    target_lon = lon.astype(np.float64)

    source_grid = _rectilinear_grid(
        da.lat.values.astype(np.float64),
        da.lon.values.astype(np.float64),
    )
    target_grid = _rectilinear_grid(target_lat, target_lon)

    weights_path = cache_dir / f"era5_conservative_{_grid_hash(source_grid, target_grid)}.nc"
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


def _cache_filename(start: str, end: str, lat: np.ndarray, lon: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(start.encode())
    h.update(end.encode())
    h.update(lat.astype(np.float32).tobytes())
    h.update(lon.astype(np.float32).tobytes())
    return f"era5_tp_{start}_{end}_{h.hexdigest()}.nc"
