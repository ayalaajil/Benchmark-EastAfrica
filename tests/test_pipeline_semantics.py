"""
Pipeline-semantics regression tests: the common land mask (does every
observational reference see the same cells?), forecast/obs date alignment
(does gather_pairs actually pair init + lead_day with the right valid date?),
and season filtering (does it partition cases correctly by valid date?).

These use small synthetic xarray objects built to the same conventions as the
real pipeline (see benchmark_ea.verification.data / scores docstrings), not
real predictions or observations, so they run instantly with no data
dependency.
"""

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea.verification.data import build_lookup_dicts
from benchmark_ea.verification.mask import apply_land_mask, land_mask
from benchmark_ea.verification.scores import gather_pairs
from benchmark_ea.verification.seasons import SEASONS, season_of


# ── Common land mask ──────────────────────────────────────────────────────────

def _synthetic_chirps(n_days=10):
    """(time, lat, lon) field with a fixed ocean/land pattern: lon index 0
    is NaN every day (ocean), the rest are always finite (land)."""
    time = pd.date_range("2024-03-01", periods=n_days, freq="D")
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 10, size=(n_days, len(lat), len(lon)))
    data[:, :, 0] = np.nan                     # ocean column, every day
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": lat, "lon": lon})


def test_land_mask_matches_chirps_validity_pattern():
    chirps_da = _synthetic_chirps()
    mask = land_mask(chirps_da, valid_frac=0.99)
    # ocean column (lon index 0) excluded, both land columns included
    assert not mask[:, 0].any()
    assert mask[:, 1:].all()
    assert mask.sum() == mask.size - mask.shape[0]


def test_build_lookup_dicts_applies_common_mask_to_era5_too():
    chirps_da = _synthetic_chirps()
    # ERA5-like: a reanalysis with real values *everywhere*, including the
    # cell CHIRPS treats as ocean — the exact failure mode the common mask
    # fixes (ERA5-verified scores silently pooling ocean cells CHIRPS never
    # sees).
    rng = np.random.default_rng(1)
    era5_da = xr.DataArray(
        rng.uniform(0, 10, size=chirps_da.shape),
        dims=chirps_da.dims, coords=chirps_da.coords)
    tamsat_da = era5_da.copy()

    chirps_2d, era5_2d, tamsat_2d, *_ = build_lookup_dicts(chirps_da, era5_da, tamsat_da)

    for d in chirps_2d:
        c_nan = np.isnan(chirps_2d[d])
        e_nan = np.isnan(era5_2d[d])
        t_nan = np.isnan(tamsat_2d[d])
        # Same cells masked out for every reference, even though the raw
        # ERA5/TAMSAT fields had no NaNs of their own at all.
        assert np.array_equal(c_nan, e_nan)
        assert np.array_equal(c_nan, t_nan)
        assert np.isnan(era5_2d[d][:, 0]).all()      # ocean column masked
        assert not np.isnan(era5_2d[d][:, 1:]).any()  # land columns intact


def test_apply_land_mask_is_a_pure_nan_overlay():
    mask = np.array([[True, False], [False, True]])
    d = {"a": np.array([[1.0, 2.0], [3.0, 4.0]])}
    out = apply_land_mask(d, mask)
    assert np.array_equal(np.isnan(out["a"]), ~mask)
    assert out["a"][0, 0] == 1.0 and out["a"][1, 1] == 4.0


# ── Forecast/obs date alignment ───────────────────────────────────────────────

def _synthetic_preds_and_obs(init_dates, lead_days):
    """A single-cell 'model' whose forecast value at (init, lead_day) is
    exactly the ordinal of the valid date (init + lead_day) — so a correct
    gather must return fc == obs for every pair, and any date-arithmetic bug
    (off-by-one, wrong lead applied) shows up as a mismatch or a missing
    case rather than silently passing."""
    lat, lon = np.array([0.0]), np.array([0.0])
    data = np.empty((len(init_dates), 1, len(lead_days), 1, 1))
    for i, init in enumerate(init_dates):
        for j, ld in enumerate(lead_days):
            vd = (init + pd.Timedelta(days=ld)).date()
            data[i, 0, j, 0, 0] = vd.toordinal()
    fc_da = xr.DataArray(
        data, dims=("init_time", "sample", "lead_day", "lat", "lon"),
        coords={"init_time": init_dates, "lead_day": lead_days,
               "lat": lat, "lon": lon})

    all_valid_dates = {(init + pd.Timedelta(days=ld)).date()
                       for init in init_dates for ld in lead_days}
    obs_2d = {d: np.array([[float(d.toordinal())]]) for d in all_valid_dates}
    return {"model": fc_da}, obs_2d


def test_gather_pairs_valid_date_alignment():
    init_dates = pd.date_range("2024-01-01", periods=20, freq="D")
    lead_days = [1, 3, 5, 7]
    preds, obs_2d = _synthetic_preds_and_obs(init_dates, lead_days)

    for ld in lead_days:
        fc_ens, obs = gather_pairs(preds, "model", obs_2d, init_dates, ld)
        assert fc_ens is not None
        assert len(obs) == len(init_dates)          # every init has a match
        assert np.allclose(fc_ens[:, 0], obs)        # fc encodes the same date as obs


def test_gather_pairs_season_filter_partitions_by_valid_date():
    # Span a full year so every season is represented.
    init_dates = pd.date_range("2024-01-01", "2024-12-20", freq="D")
    lead_days = [1, 3, 5, 7]
    preds, obs_2d = _synthetic_preds_and_obs(init_dates, lead_days)
    ld = 3

    fc_all, obs_all = gather_pairs(preds, "model", obs_2d, init_dates, ld)
    total = len(obs_all)

    counted = 0
    for season in SEASONS[1:]:                       # skip "annual"
        fc_s, obs_s = gather_pairs(preds, "model", obs_2d, init_dates, ld, season=season)
        if fc_s is None:
            continue
        counted += len(obs_s)
        for ordinal in obs_s:
            d = date.fromordinal(int(ordinal))
            assert season_of(d) == season

    assert counted == total       # the four seasons partition the full set
