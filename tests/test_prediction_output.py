"""
Unit tests for the inference output shaping/regridding plumbing.

These exercise the pure data-handling paths (no model, no GPU) that turn raw
model output into the canonical East Africa zarr layout:
  * benchmark_ea.regrid.to_ea_grid          — bbox subset + interp to the 1° grid
  * benchmark_ea.regrid.extra_vars_to_canonical — snapshot-per-lead + member scope
  * benchmark_ea.models.base.ModelAdapter.assemble_output — precip vs all-variables
"""

import numpy as np
import pytest
import xarray as xr

from benchmark_ea.config import BenchmarkConfig
from benchmark_ea import regrid
from benchmark_ea.models.base import ModelAdapter


def _global_da(values_fn, descending_lat=True):
    """A 1°-spaced field covering East Africa with margin (aligned to EA grid)."""
    lat = np.arange(18, -19, -1.0) if descending_lat else np.arange(-18, 19, 1.0)
    lon = np.arange(22, 59, 1.0)
    data = values_fn(lat[:, None], lon[None, :]).astype(np.float32)
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def _raw_dataset(n_sample=None):
    """Raw model output: temperature (level) + u10 over a 7-day, 1-day-step axis."""
    time = np.array([np.timedelta64(d * 24, "h") for d in range(1, 8)])
    lat = np.arange(18, -19, -1.0)
    lon = np.arange(22, 59, 1.0)
    level = np.array([500, 850])

    # value encodes the lead day so we can assert correct snapshot selection
    lead_idx = np.arange(1, 8)
    temp = np.zeros((len(time), len(level), len(lat), len(lon)), dtype=np.float32)
    temp += lead_idx[:, None, None, None]            # = lead day
    u10 = np.zeros((len(time), len(lat), len(lon)), dtype=np.float32)
    u10 += 10 * lead_idx[:, None, None]              # = 10 * lead day

    ds = xr.Dataset(
        {
            "temperature": (["time", "level", "lat", "lon"], temp),
            "u10": (["time", "lat", "lon"], u10),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )
    if n_sample is not None:
        # broadcast identical members so the ensemble mean is unchanged
        ds = ds.expand_dims({"sample": np.arange(n_sample)})
    return ds


def test_to_ea_grid_subsets_and_regrids_to_canonical_grid():
    cfg = BenchmarkConfig()
    da = _global_da(lambda la, lo: la + lo)          # smooth linear field
    out = regrid.to_ea_grid(da, cfg)

    assert list(out.lat.values) == pytest.approx(list(cfg.lat_vals))
    assert list(out.lon.values) == pytest.approx(list(cfg.lon_vals))
    assert out.sizes == {"lat": len(cfg.lat_vals), "lon": len(cfg.lon_vals)}
    # linear field is reproduced exactly by bilinear interpolation
    expected = cfg.lat_vals[:, None] + cfg.lon_vals[None, :]
    np.testing.assert_allclose(out.values, expected, atol=1e-3)


def test_extra_vars_to_canonical_deterministic_shape_and_snapshot():
    cfg = BenchmarkConfig(lead_days=[1, 3, 5, 7])
    raw = _raw_dataset()
    out = regrid.extra_vars_to_canonical(
        raw, np.datetime64("2024-03-01"), cfg, sample_dim=None
    )

    assert out.sizes["init_time"] == 1
    assert out.sizes["sample"] == 1
    assert list(out.lead_day.values) == cfg.lead_days
    assert out.sizes["lat"] == len(cfg.lat_vals)
    assert out.sizes["lon"] == len(cfg.lon_vals)
    assert out.sizes["level"] == 2
    # each variable was sampled at the matching lead-day snapshot
    for i, lead in enumerate(cfg.lead_days):
        np.testing.assert_allclose(out["u10"].isel(init_time=0, sample=0,
                                                   lead_day=i).values, 10 * lead,
                                   atol=1e-3)
        np.testing.assert_allclose(out["temperature"].isel(init_time=0, sample=0,
                                                           lead_day=i).values, lead,
                                   atol=1e-3)


def test_extra_vars_member_scope_mean_vs_all():
    cfg_mean = BenchmarkConfig(extra_var_members="mean")
    cfg_all = BenchmarkConfig(extra_var_members="all")
    raw = _raw_dataset(n_sample=4)

    out_mean = regrid.extra_vars_to_canonical(
        raw, np.datetime64("2024-03-01"), cfg_mean, sample_dim="sample")
    out_all = regrid.extra_vars_to_canonical(
        raw, np.datetime64("2024-03-01"), cfg_all, sample_dim="sample")

    assert out_mean.sizes["sample"] == 1
    assert out_all.sizes["sample"] == 4


def test_assemble_output_precip_only_passthrough():
    cfg = BenchmarkConfig(save_variables="precip")
    canonical = xr.Dataset(
        {"total_precipitation": (["init_time", "sample", "lead_day", "lat", "lon"],
                                 np.ones((1, 1, 4, 3, 2), dtype=np.float32))}
    )
    out = ModelAdapter.assemble_output(canonical, _raw_dataset(), "2024-03-01", cfg)
    assert set(out.data_vars) == {"total_precipitation"}


def test_assemble_output_all_merges_extra_vars():
    cfg = BenchmarkConfig(save_variables="all", lead_days=[1, 3, 5, 7])
    canonical = xr.Dataset(
        {
            "total_precipitation": xr.DataArray(
                np.ones((1, 1, 4, len(cfg.lat_vals), len(cfg.lon_vals)), np.float32),
                dims=["init_time", "sample", "lead_day", "lat", "lon"],
                coords={
                    "init_time": [np.datetime64("2024-03-01", "ns")],
                    "sample": [0], "lead_day": cfg.lead_days,
                    "lat": cfg.lat_vals, "lon": cfg.lon_vals,
                },
            )
        }
    )
    out = ModelAdapter.assemble_output(
        canonical, _raw_dataset(), np.datetime64("2024-03-01"), cfg,
        sample_dim=None,
    )
    assert "total_precipitation" in out.data_vars
    assert "temperature" in out.data_vars
    assert "u10" in out.data_vars


def test_config_validates_save_options():
    with pytest.raises(ValueError):
        BenchmarkConfig(save_variables="bogus")
    with pytest.raises(ValueError):
        BenchmarkConfig(extra_var_members="bogus")


# ── Conservative regridding (shared model/obs precip operator) ────────────────

pytest.importorskip("xesmf")


def _fine_precip_da(cfg):
    """A positive 0.25° 'rainfall' field whose outer cell bounds tile the target
    grid's footprint exactly (lat descending, like FourCastNet/CHIRPS)."""
    lo_lat = float(cfg.lat_vals.min()) - 0.5 + 0.125   # first cell centre
    hi_lat = float(cfg.lat_vals.max()) + 0.5 - 0.125
    lo_lon = float(cfg.lon_vals.min()) - 0.5 + 0.125
    hi_lon = float(cfg.lon_vals.max()) + 0.5 - 0.125
    lat = np.arange(hi_lat, lo_lat - 1e-6, -0.25)       # descending
    lon = np.arange(lo_lon, hi_lon + 1e-6, 0.25)
    rng = np.random.default_rng(0)
    data = rng.gamma(shape=1.5, scale=3.0, size=(lat.size, lon.size)).astype(np.float32)
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def test_conservative_regrid_conserves_area_weighted_mean(tmp_path):
    """Coarsening 0.25°→1° must preserve the cos(lat)-weighted domain mean
    (mass), unlike bilinear point-sampling. Source footprint == target footprint,
    so the two area-weighted means must agree."""
    cfg = BenchmarkConfig()
    da = _fine_precip_da(cfg)

    out = regrid.conservative_regrid(
        da, cfg.lat_vals, cfg.lon_vals, tmp_path, tag="test"
    )

    assert list(out.lat.values) == pytest.approx(list(cfg.lat_vals))
    assert list(out.lon.values) == pytest.approx(list(cfg.lon_vals))

    src_mean = float(da.weighted(np.cos(np.deg2rad(da.lat))).mean().values)
    out_mean = float(out.weighted(np.cos(np.deg2rad(out.lat))).mean().values)
    assert out_mean == pytest.approx(src_mean, rel=0.01)


def test_conservative_regrid_normalises_latlon_names_and_orientation(tmp_path):
    """latitude/longitude names and a descending axis are handled transparently."""
    cfg = BenchmarkConfig()
    da = _fine_precip_da(cfg).rename({"lat": "latitude", "lon": "longitude"})

    out = regrid.conservative_regrid(
        da, cfg.lat_vals, cfg.lon_vals, tmp_path, tag="test"
    )
    assert out.sizes == {"lat": len(cfg.lat_vals), "lon": len(cfg.lon_vals)}
    assert float(out.lat[0]) < float(out.lat[-1])          # ascending
    assert bool(np.isfinite(out).all())                    # every cell covered
