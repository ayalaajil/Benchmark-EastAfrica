"""
East Africa spatial masks.

Uses Natural Earth 50m polygons via cartopy — the same approach as AIM-for-Scale
so masks are consistent across projects.

Key additions vs AIM-for-Scale:
  - Rwanda and Burundi included alongside the five flood-season countries
  - lat/cos-weighted area weights for unbiased regional averages
"""

import numpy as np
import xarray as xr

# Country → hex colour (used consistently in all comparison plots)
COUNTRIES: dict[str, str] = {
    "Kenya":    "#E53935",
    "Ethiopia": "#FB8C00",
    "Tanzania": "#43A047",
    "Somalia":  "#1E88E5",
    "Uganda":   "#8E24AA",
    "Rwanda":   "#00897B",
    "Burundi":  "#F4511E",
}


def land_mask(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """
    Rasterise Natural Earth 50m land polygons onto the given 1° grid.

    Returns float32 DataArray (lat, lon): 1.0 = land, 0.0 = ocean / lake.
    """
    import cartopy.io.shapereader as shpreader
    import shapely
    from shapely.ops import unary_union

    shpfile   = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    land_geom = unary_union(list(shpreader.Reader(shpfile).geometries()))

    lons_2d, lats_2d = np.meshgrid(lon, lat)
    inside = shapely.contains_xy(land_geom, lons_2d.ravel(), lats_2d.ravel())

    return xr.DataArray(
        inside.reshape(lats_2d.shape).astype(np.float32),
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        attrs={"description": "Natural Earth 50m land mask: 1=land, 0=ocean"},
    )


def country_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, xr.DataArray]:
    """
    Per-country boolean masks using Natural Earth 50m admin boundaries.

    Returns dict: country name → DataArray(bool, dims=[lat, lon]).
    """
    import cartopy.io.shapereader as shpreader
    import shapely

    shpfile = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfile)

    geoms: dict = {}
    for rec in reader.records():
        name = rec.attributes.get("NAME_EN") or rec.attributes.get("NAME", "")
        if name in COUNTRIES:
            geoms[name] = rec.geometry

    lons_2d, lats_2d = np.meshgrid(lon, lat)
    flat_lon, flat_lat = lons_2d.ravel(), lats_2d.ravel()

    return {
        name: xr.DataArray(
            shapely.contains_xy(geom, flat_lon, flat_lat).reshape(lats_2d.shape),
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"],
        )
        for name, geom in geoms.items()
    }


def area_weights(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """
    cos(lat) area weights normalised so they sum to 1 over the full grid.
    Used for spatially-fair regional averages.
    """
    w = np.cos(np.deg2rad(lat))
    weights = xr.DataArray(
        np.outer(w, np.ones_like(lon)).astype(np.float32),
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
    )
    return weights / weights.sum()
