"""
Probabilistic and deterministic verification metrics — single source of truth.

These are the metric implementations used by ``run_verification.py`` and the
standalone analysis scripts. Import them from here rather than redefining them,
so every entry point scores identically, e.g.::

    from benchmark_ea.metrics import crps_ensemble, rank_histogram

Array conventions used throughout:
  fc_ens   : (case, member, ...)  ensemble forecast
  fc_mean  : (case, ...)          ensemble mean / deterministic forecast
  obs      : (case, ...)          verifying observation
  weights  : (case,)              optional per-case aggregation weight (e.g.
                                   cos(latitude) area weight); unweighted
                                   (equal-weight) aggregation when omitted.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

# 95% two-sided normal quantile — for Wilson score intervals on reliability freqs.
_Z95 = _norm.ppf(0.975)


# ── Canonical verification metrics ────────────────────────────────────────────

def deterministic_metrics(fc_mean, obs, weights=None):
    """Bias, MAE, RMSE and the pooled Pearson correlation of a deterministic
    (or ensemble-mean) forecast against observations, optionally area-weighted.

    ``pooled_corr`` is the correlation over the pooled (case x cell) sample —
    not a per-case spatial correlation — hence the name.
    """
    fc_mean = np.asarray(fc_mean, dtype=float)
    obs     = np.asarray(obs, dtype=float)
    mask    = np.isfinite(fc_mean) & np.isfinite(obs)
    fc_mean, obs = fc_mean[mask], obs[mask]
    w = None if weights is None else np.asarray(weights, dtype=float)[mask]

    err = fc_mean - obs
    bias = float(np.average(err, weights=w))
    mae  = float(np.average(np.abs(err), weights=w))
    rmse = float(np.sqrt(np.average(err ** 2, weights=w)))

    if len(obs) > 1 and np.std(fc_mean) > 0 and np.std(obs) > 0:
        if w is None:
            pooled_corr = float(np.corrcoef(fc_mean, obs)[0, 1])
        else:
            wf, wo = np.average(fc_mean, weights=w), np.average(obs, weights=w)
            cov   = np.average((fc_mean - wf) * (obs - wo), weights=w)
            var_f = np.average((fc_mean - wf) ** 2, weights=w)
            var_o = np.average((obs - wo) ** 2, weights=w)
            pooled_corr = (float(cov / np.sqrt(var_f * var_o))
                           if var_f > 0 and var_o > 0 else np.nan)
    else:
        pooled_corr = np.nan

    return {
        "n":        int(mask.sum()),
        "obs_mean": float(np.average(obs, weights=w)),
        "fc_mean":  float(np.average(fc_mean, weights=w)),
        "bias":     bias,
        "mae":      mae,
        "rmse":     rmse,
        "pooled_corr": pooled_corr,
    }


def contingency_scores(fc_mean, obs, threshold):
    """Standard 2x2 contingency-table scores (POD, FAR, CSI, frequency bias)
    at one exceedance threshold.

    Hit/miss/false-alarm/correct-negative counts are raw, unweighted cell
    counts — contingency scores are ratios of counts, and an area-weighted
    count is not a meaningful quantity (weight the *rate* fields downstream
    if an area-weighted event frequency is needed; this function does not).
    """
    fc_e = np.asarray(fc_mean) > threshold
    ob_e = np.asarray(obs)     > threshold
    mask = np.isfinite(fc_mean) & np.isfinite(obs)
    fc_e, ob_e = fc_e[mask], ob_e[mask]
    hits  = np.sum( fc_e &  ob_e)
    miss  = np.sum(~fc_e &  ob_e)
    fa    = np.sum( fc_e & ~ob_e)
    cn    = np.sum(~fc_e & ~ob_e)
    return {
        "hits": int(hits), "misses": int(miss),
        "false_alarms": int(fa), "correct_negatives": int(cn),
        "pod":           hits / (hits + miss) if (hits + miss) > 0 else np.nan,
        "far":           fa   / (hits + fa)   if (hits + fa)   > 0 else np.nan,
        "csi":           hits / (hits + miss + fa) if (hits + miss + fa) > 0 else np.nan,
        "frequency_bias":(hits + fa) / (hits + miss) if (hits + miss) > 0 else np.nan,
        "observed_event_rate":  float(np.mean(ob_e)),
        "forecast_event_rate":  float(np.mean(fc_e)),
    }


def crps_ensemble(fc_ens, obs, fair=True):
    """Continuous ranked probability score per case, (M-member) ensemble vs
    observation.

        CRPS = E|X - y| - c * E|X - X'|,   X, X' iid draws from the ensemble

    With ``fair=True`` (default), the second term uses the Ferro et al.
    (2014) finite-ensemble-unbiased estimator (pairs without replacement,
    i != j, divisor M(M-1)); ``fair=False`` gives the standard plug-in
    estimator (pairs with replacement, divisor M^2), which is what this
    function computed prior to the fair-CRPS fix. For M == 1 (deterministic
    forecasts) both reduce exactly to the mean absolute error.

    fc_ens : (case, member)   obs : (case,)   → returns (case,) CRPS values.
    """
    fc_ens = np.asarray(fc_ens, dtype=float)
    obs    = np.asarray(obs, dtype=float)
    m = fc_ens.shape[1]
    term1 = np.mean(np.abs(fc_ens - obs[:, None]), axis=1)
    if m <= 1:
        return term1
    # 0.5 * mean over all M^2 (i, j) pairs (including i == j, which is 0) —
    # the standard plug-in estimator's spread term.
    term2 = 0.5 * np.mean(np.abs(fc_ens[:, :, None] - fc_ens[:, None, :]), axis=(1, 2))
    if fair:
        # Rescale to the i != j, divide-by-M(M-1) fair estimator: the M^2-pair
        # mean above equals (M(M-1)/M^2) times the fair i!=j mean, so
        # multiplying by M/(M-1) recovers it exactly (see Ferro et al. 2014).
        term2 = term2 * m / (m - 1)
    return term1 - term2


def brier_score_ensemble(fc_ens, obs, threshold, weights=None):
    """Brier score of the ensemble exceedance probability at one threshold,
    optionally area-weighted."""
    prob  = (fc_ens > threshold).mean(axis=1)
    event = (obs > threshold).astype(float)
    return float(np.average((prob - event) ** 2, weights=weights))


def interval_coverage(fc_ens, obs, nominal, weights=None):
    """Empirical coverage and mean width of the nominal central prediction
    interval (e.g. nominal=0.80 → the 10th-90th ensemble percentile band),
    optionally area-weighted."""
    alpha  = 1 - nominal
    lower  = np.quantile(fc_ens, alpha / 2,     axis=1)
    upper  = np.quantile(fc_ens, 1 - alpha / 2, axis=1)
    covered = (obs >= lower) & (obs <= upper)
    return {
        "nominal_coverage":   nominal,
        "empirical_coverage": float(np.average(covered.astype(float), weights=weights)),
        "mean_width":         float(np.average(upper - lower, weights=weights)),
    }


def spread_skill(fc_ens, obs, weights=None):
    """Ensemble spread, RMSE of the ensemble mean, and their ratio (SSR),
    with the Fortin et al. (2014) finite-ensemble bias correction —
    the single spread/SSR definition used everywhere in this package.

        spread = sqrt( (M+1)/M * <ensemble variance> )
        SSR    = spread / RMSE(ensemble mean)          1 = well calibrated

    Uncorrected mean(member std)/RMSE (no (M+1)/M term) systematically
    understates spread for small M and must not be used instead.

    fc_ens : (case, member)   obs : (case,)   → returns (spread, rmse, ssr).
    """
    fc_ens = np.asarray(fc_ens, dtype=float)
    obs    = np.asarray(obs, dtype=float)
    m = fc_ens.shape[1]
    fc_mean = fc_ens.mean(axis=1)
    err  = fc_mean - obs
    var  = fc_ens.var(axis=1, ddof=1)
    rmse = float(np.sqrt(np.average(err ** 2, weights=weights)))
    corr = (m + 1) / m
    spread = float(np.sqrt(corr * np.average(var, weights=weights)))
    ssr = spread / rmse if rmse > 0 else np.nan
    return spread, rmse, ssr


def rank_histogram(fc_ens, obs, rng=None):
    """Talagrand rank histogram with randomized ranks for ties.

    Precipitation has many exact zeros, so counting only members *strictly*
    below the observation assigns every tied case (e.g. a dry day where members
    are also 0) to rank 0, producing a spurious left spike. Following
    Hamill (2001) we draw the observation's rank uniformly within the tied
    block, so a perfectly dry ensemble contributes flat rather than to rank 0.

    Returns the (member + 1,)-length array of relative frequencies (sums to 1).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs     = np.asarray(obs)[:, None]
    n_below = (fc_ens < obs).sum(axis=1)                       # strictly below
    n_tied  = (fc_ens == obs).sum(axis=1)                      # exact ties
    # uniform integer offset in [0, n_tied] within the tied block
    ranks   = n_below + np.floor(rng.random(len(n_below)) * (n_tied + 1)).astype(int)
    n_bins  = fc_ens.shape[1] + 1
    counts  = np.bincount(ranks, minlength=n_bins)
    return counts / counts.sum()


def rank_histogram_flatness(rel_freq):
    """Flatness statistic for a rank histogram's relative frequencies
    (as returned by ``rank_histogram``): normalized standard deviation
    around the uniform expectation 1/n_bins.

        delta = std(rel_freq) / (1 / n_bins)

    0 = perfectly flat (well calibrated). Large values indicate a markedly
    non-flat histogram (U-shaped under-dispersion, dome-shaped
    over-dispersion, or a skew indicating systematic bias) — the shape itself
    (not this scalar) says which.
    """
    rel_freq = np.asarray(rel_freq, dtype=float)
    n_bins = len(rel_freq)
    if n_bins == 0 or not np.isfinite(rel_freq).all():
        return np.nan
    expected = 1.0 / n_bins
    return float(np.std(rel_freq) / expected)


def reliability_diagram(fc_ens, obs, threshold):
    """Reliability diagram at one threshold, binned by the exact discrete
    ensemble-exceedance fraction (0/M, 1/M, ..., M/M — the only probability
    values an M-member ensemble can produce).

    Returns (prob_levels, obs_freq, counts), each length M + 1.
    """
    n_members   = fc_ens.shape[1]
    prob_fc     = (fc_ens > threshold).sum(axis=1) / n_members
    bin_obs     = (obs > threshold).astype(float)
    prob_levels = np.arange(n_members + 1) / n_members
    obs_freq, counts = [], []
    for p in prob_levels:
        mask = np.isclose(prob_fc, p)
        counts.append(mask.sum())
        obs_freq.append(bin_obs[mask].mean() if mask.sum() > 0 else np.nan)
    return prob_levels, np.array(obs_freq), np.array(counts)


def brier_decomposition(fc_ens, obs, threshold):
    """Murphy (1973) decomposition of the Brier score into reliability,
    resolution, and uncertainty, using the same discrete probability bins as
    ``reliability_diagram`` (so REL/RES/UNC are computed on an exact
    partition of the pooled sample, and the identity below holds exactly,
    not just approximately):

        BS  = REL - RES + UNC
        BSS = 1 - BS / UNC            (UNC = the base-rate reference Brier)

    Returns a dict with bs, bss, reliability, resolution, uncertainty.
    """
    prob_levels, obs_freq, counts = reliability_diagram(fc_ens, obs, threshold)
    obs_b = (np.asarray(obs) > threshold).astype(float)
    n_tot = counts.sum()
    if n_tot == 0:
        return dict(bs=np.nan, bss=np.nan, reliability=np.nan,
                    resolution=np.nan, uncertainty=np.nan)

    clim = float(obs_b.mean())
    finite = np.isfinite(obs_freq)
    rel = float(np.sum(counts[finite] * (prob_levels[finite] - obs_freq[finite]) ** 2) / n_tot)
    res = float(np.sum(counts[finite] * (obs_freq[finite] - clim) ** 2) / n_tot)
    unc = float(clim * (1.0 - clim))
    bs  = rel - res + unc
    bss = float(1.0 - bs / unc) if unc > 1e-9 else np.nan
    return dict(bs=bs, bss=bss, reliability=rel, resolution=res, uncertainty=unc)


def reliability_diagram_local(fc_ens, obs_arr, thresh_arr, n_bins=10):
    n_members = fc_ens.shape[1]
    prob_fc   = (fc_ens > thresh_arr[:, None]).sum(axis=1) / n_members
    bin_obs   = (obs_arr > thresh_arr).astype(float)
    edges     = np.linspace(0, 1, n_bins + 1)
    prob_levels, obs_freq, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob_fc >= lo) & (prob_fc < hi) if lo > 0 else (prob_fc >= 0) & (prob_fc <= hi)
        ct   = int(mask.sum())
        counts.append(ct)
        prob_levels.append(float((lo + hi) / 2))
        obs_freq.append(float(bin_obs[mask].mean()) if ct > 0 else np.nan)
    return np.array(prob_levels), np.array(obs_freq), np.array(counts)


def wilson_ci(counts, obs_freq, z=_Z95):
    n = np.asarray(counts, dtype=float)
    p = np.asarray(obs_freq, dtype=float)
    lo, hi = np.full_like(p, np.nan), np.full_like(p, np.nan)
    ok = (n > 0) & ~np.isnan(p)
    nv, pv = n[ok], p[ok]
    denom  = 1 + z ** 2 / nv
    centre = (pv + z ** 2 / (2 * nv)) / denom
    margin = z * np.sqrt(pv * (1 - pv) / nv + z ** 2 / (4 * nv ** 2)) / denom
    lo[ok] = np.clip(centre - margin, 0, 1)
    hi[ok] = np.clip(centre + margin, 0, 1)
    return lo, hi


def compute_ece(prob_levels, obs_freq, counts):
    total = counts.sum()
    if total == 0:
        return np.nan
    return float(np.nansum(counts / total * np.abs(np.asarray(obs_freq) - np.asarray(prob_levels))))
