"""
Analytic / hand-computed checks for benchmark_ea.metrics — the single source
of truth for every score in the verification pipeline. These are pure-math
tests (no zarr, no network): each one either has a closed-form answer derived
independently of the implementation, or a synthetic construction whose
correct value is known by design.
"""

import numpy as np

from benchmark_ea.metrics import (
    brier_decomposition,
    brier_score_ensemble,
    compute_ece,
    crps_ensemble,
    deterministic_metrics,
    interval_coverage,
    rank_histogram,
    rank_histogram_flatness,
    reliability_diagram,
    spread_skill,
)


# ── Fair CRPS (Ferro et al. 2014) ─────────────────────────────────────────────

def test_crps_fair_hand_case_m2():
    # members [0, 2], obs = 1: fair CRPS has the closed form
    #   term1 - (1/(2*M*(M-1))) * sum_{i!=j}|xi-xj|
    #   = 1 - (1/4)*(|0-2|+|2-0|) = 1 - 1 = 0
    fc = np.array([[0.0, 2.0]])
    obs = np.array([1.0])
    fair = crps_ensemble(fc, obs, fair=True)
    standard = crps_ensemble(fc, obs, fair=False)
    assert np.isclose(fair[0], 0.0, atol=1e-12)
    assert np.isclose(standard[0], 0.5, atol=1e-12)
    assert fair[0] < standard[0]


def test_crps_fair_hand_case_m3():
    # members [0, 1, 5], obs = 2.
    fc = np.array([[0.0, 1.0, 5.0]])
    obs = np.array([2.0])
    fair = crps_ensemble(fc, obs, fair=True)[0]
    standard = crps_ensemble(fc, obs, fair=False)[0]
    # hand-derived: term1 = mean(2,1,3) = 2
    # fair term2   = 0.5 * (1/6) * sum_{i!=j}|xi-xj| = 0.5 * 20/6 = 5/3
    # standard term2 = 0.5 * (1/9) * sum_{all i,j}|xi-xj| = 0.5 * 20/9 = 10/9
    assert np.isclose(fair, 2.0 - 5.0 / 3.0, atol=1e-9)
    assert np.isclose(standard, 2.0 - 10.0 / 9.0, atol=1e-9)
    assert fair < standard


def test_crps_single_member_equals_mae():
    fc = np.array([[5.0], [1.0], [-3.0]])
    obs = np.array([2.0, 1.0, 0.0])
    fair = crps_ensemble(fc, obs, fair=True)
    standard = crps_ensemble(fc, obs, fair=False)
    mae = np.abs(fc[:, 0] - obs)
    assert np.allclose(fair, mae)
    assert np.allclose(standard, mae)


def test_crps_fair_matches_closed_form_for_uniform_forecast():
    # For X, X' iid ~ Uniform(0, 1) and a scalar observation y in [0, 1], the
    # true (population) CRPS of the Uniform(0,1) forecast distribution has
    # the closed form CRPS(y) = (1/3) * (y^3 + (1-y)^3) (derived from
    # CRPS(F,y) = int (F(x) - 1{x>=y})^2 dx). The fair estimator, averaged
    # over many independent M-member samples from that same distribution, is
    # an unbiased estimator of this population value; the plug-in (standard)
    # estimator is biased at finite M and should sit further from it.
    rng = np.random.default_rng(0)
    y = 0.3
    closed_form = (y ** 3 + (1 - y) ** 3) / 3.0

    m, n_cases = 10, 40_000
    fc = rng.uniform(0.0, 1.0, size=(n_cases, m))
    obs = np.full(n_cases, y)

    fair_mean = float(np.mean(crps_ensemble(fc, obs, fair=True)))
    standard_mean = float(np.mean(crps_ensemble(fc, obs, fair=False)))

    assert abs(fair_mean - closed_form) < 0.01
    assert abs(fair_mean - closed_form) < abs(standard_mean - closed_form)


# ── Spread-skill ratio (Fortin et al. 2014) ───────────────────────────────────

def test_spread_skill_fortin_correction_vs_naive():
    # obs and M ensemble members all iid ~ N(0, 1): a textbook "perfectly
    # calibrated ensemble" construction (Fortin et al. 2014). With M members,
    #   Var(ensemble mean) = 1/M,  Var(err) = 1 + 1/M
    #   E[sample var (ddof=1)] = 1
    # Fortin spread = sqrt((M+1)/M * 1) = sqrt(1 + 1/M) = RMSE  → SSR = 1
    # The naive (uncorrected) spread = sqrt(1) understates this and gives
    # SSR = 1/sqrt(1+1/M) < 1.
    rng = np.random.default_rng(1)
    m, n_cases = 5, 200_000
    obs = rng.normal(0.0, 1.0, size=n_cases)
    members = rng.normal(0.0, 1.0, size=(n_cases, m))

    spread, rmse, ssr = spread_skill(members, obs)
    assert abs(ssr - 1.0) < 0.02

    naive_spread = float(np.mean(members.std(axis=1, ddof=1)))
    naive_rmse = float(np.sqrt(np.mean((members.mean(axis=1) - obs) ** 2)))
    naive_ssr = naive_spread / naive_rmse
    assert naive_ssr < 0.97          # measurably below 1
    assert naive_ssr < ssr           # and below the corrected value


def test_spread_skill_weighted_differs_from_unweighted():
    # Two "latitude bands": band 0 has huge errors (should dominate under a
    # weight of 1) but band 1 (near-zero errors) gets a much larger weight —
    # the weighted RMSE must be pulled toward band 1's near-zero error.
    obs = np.array([0.0, 0.0, 0.0, 0.0])
    members = np.array([
        [10.0, 10.0], [-10.0, -10.0],   # band 0: huge error, weight 1
        [0.01, -0.01], [0.0, 0.0],      # band 1: tiny error, weight 1e6
    ])
    weights = np.array([1.0, 1.0, 1e6, 1e6])
    _, rmse_w, _ = spread_skill(members, obs, weights=weights)
    _, rmse_u, _ = spread_skill(members, obs, weights=None)
    # unweighted MSE = mean(100, 100, ~0, 0) = 50 → rmse_u ~ 7.07
    # weighted MSE ~ 200 / 2e6 ~ 1e-4 → rmse_w ~ 0.01, two orders smaller
    assert rmse_w < rmse_u
    assert rmse_w < 0.1


# ── Rank histograms (Hamill 2001 randomized ties) ─────────────────────────────

def test_rank_histogram_all_zero_ties_is_flat():
    # Every member and the observation are exactly 0: without randomized
    # ties every case would land in rank 0 (the historical bug this guards
    # against); with Hamill (2001) randomization it must be flat.
    m, n_cases = 4, 50_000
    fc = np.zeros((n_cases, m))
    obs = np.zeros(n_cases)
    freq = rank_histogram(fc, obs, rng=np.random.default_rng(2))
    assert len(freq) == m + 1
    assert np.allclose(freq, 1.0 / (m + 1), atol=0.01)
    assert rank_histogram_flatness(freq) < 0.05


def test_rank_histogram_exchangeable_draws_is_flat():
    # obs and the M members are all iid draws from the same distribution
    # ("exchangeable") — the textbook calibrated-ensemble construction —
    # so the rank of the observation among the M+1 values is uniform.
    rng = np.random.default_rng(3)
    m, n_cases = 6, 60_000
    pool = rng.normal(size=(n_cases, m + 1))
    obs = pool[:, 0]
    fc = pool[:, 1:]
    freq = rank_histogram(fc, obs, rng=np.random.default_rng(4))
    assert np.allclose(freq, 1.0 / (m + 1), atol=0.01)
    assert rank_histogram_flatness(freq) < 0.05


# ── Brier score decomposition (Murphy 1973) ───────────────────────────────────

def test_brier_decomposition_identity_and_hand_case():
    # 4 cases, M = 4 members, threshold = 0.5. Every case has exactly one
    # member above the threshold (forecast probability always 0.25); only
    # the first case's observation exceeds the threshold, so the event rate
    # (= climatology) is also 0.25 — forecast probability equals the
    # observed frequency in the only populated bin, and equals climatology,
    # so REL = RES = 0 and BS = UNC exactly (BSS = 0: a base-rate forecast).
    fc = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    obs = np.array([1.0, 0.0, 0.0, 0.0])
    dec = brier_decomposition(fc, obs, threshold=0.5)

    assert np.isclose(dec["reliability"], 0.0, atol=1e-9)
    assert np.isclose(dec["resolution"], 0.0, atol=1e-9)
    assert np.isclose(dec["uncertainty"], 0.25 * 0.75, atol=1e-9)
    assert np.isclose(dec["bs"], dec["reliability"] - dec["resolution"] + dec["uncertainty"])
    assert np.isclose(dec["bss"], 0.0, atol=1e-9)
    # BS must also equal the direct Brier score of the same forecast/obs.
    assert np.isclose(dec["bs"], brier_score_ensemble(fc, obs, threshold=0.5))


def test_brier_decomposition_identity_random_case():
    # A less contrived case: the algebraic identity BS = REL - RES + UNC must
    # hold regardless of how skillful the forecast is.
    rng = np.random.default_rng(5)
    n_cases, m = 500, 8
    fc = rng.uniform(0, 1, size=(n_cases, m))
    obs = rng.uniform(0, 1, size=n_cases)
    dec = brier_decomposition(fc, obs, threshold=0.5)
    identity = dec["reliability"] - dec["resolution"] + dec["uncertainty"]
    assert np.isclose(dec["bs"], identity, atol=1e-9)


# ── Interval coverage ──────────────────────────────────────────────────────────

def test_interval_coverage_exact_quantile_grid():
    # A 101-member "ensemble" that is exactly the 0, 0.01, ..., 1.0 quantile
    # grid of Uniform(0,1); obs fixed at the median. The 80% interval is then
    # exactly [0.1, 0.9] by construction, which always contains 0.5.
    n_cases = 50
    grid = np.linspace(0.0, 1.0, 101)
    fc = np.tile(grid, (n_cases, 1))
    obs = np.full(n_cases, 0.5)
    row = interval_coverage(fc, obs, nominal=0.80)
    assert np.isclose(row["empirical_coverage"], 1.0)
    assert np.isclose(row["mean_width"], 0.8, atol=1e-9)


# ── Reliability diagram + ECE ──────────────────────────────────────────────────

def test_reliability_diagram_and_ece_hand_case():
    # Reuse the perfectly-calibrated single-bin construction above: forecast
    # probability (0.25) exactly matches the observed frequency in the only
    # populated bin, so the calibration gap — and therefore ECE — is exactly 0.
    fc = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    obs = np.array([1.0, 0.0, 0.0, 0.0])
    prob_levels, obs_freq, counts = reliability_diagram(fc, obs, threshold=0.5)
    assert counts.sum() == 4
    assert counts[1] == 4                      # all four cases in the p=0.25 bin
    assert np.isclose(obs_freq[1], 0.25)
    ece = compute_ece(prob_levels, obs_freq, counts)
    assert np.isclose(ece, 0.0, atol=1e-9)


# ── Weighted deterministic metrics ────────────────────────────────────────────

def test_deterministic_metrics_weighted_differs_from_unweighted():
    # Two groups of forecasts/obs with very different error, weighted so the
    # low-error group dominates: the weighted bias/RMSE must sit near the
    # low-error group's values, not the simple average.
    fc  = np.array([10.0, 10.0, 0.0, 0.0])
    obs = np.array([0.0,  0.0,  0.0, 0.0])
    weights = np.array([1.0, 1.0, 100.0, 100.0])
    weighted = deterministic_metrics(fc, obs, weights=weights)
    unweighted = deterministic_metrics(fc, obs, weights=None)
    assert weighted["rmse"] < unweighted["rmse"]
    assert weighted["bias"] < unweighted["bias"]
    assert weighted["rmse"] < 1.0
