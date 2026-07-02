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
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

# 95% two-sided normal quantile — for Wilson score intervals on reliability freqs.
_Z95 = _norm.ppf(0.975)


# ── Canonical verification metrics ────────────────────────────────────────────

def deterministic_metrics(fc_mean, obs):
    fc_mean = np.asarray(fc_mean, dtype=float)
    obs     = np.asarray(obs, dtype=float)
    mask    = np.isfinite(fc_mean) & np.isfinite(obs)
    fc_mean, obs = fc_mean[mask], obs[mask]
    err = fc_mean - obs
    out = {
        "n":        int(mask.sum()),
        "obs_mean": float(np.mean(obs)),
        "fc_mean":  float(np.mean(fc_mean)),
        "bias":     float(np.mean(err)),
        "mae":      float(np.mean(np.abs(err))),
        "rmse":     float(np.sqrt(np.mean(err ** 2))),
        "corr":     float(np.corrcoef(fc_mean, obs)[0, 1])
                    if len(obs) > 1 and np.std(fc_mean) > 0 and np.std(obs) > 0
                    else np.nan,
    }
    return out


def contingency_scores(fc_mean, obs, threshold):
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


def crps_ensemble(fc_ens, obs):
    term1 = np.mean(np.abs(fc_ens - obs[:, None]), axis=1)
    term2 = 0.5 * np.mean(np.abs(fc_ens[:, :, None] - fc_ens[:, None, :]), axis=(1, 2))
    return term1 - term2


def brier_score_ensemble(fc_ens, obs, threshold):
    prob  = (fc_ens > threshold).mean(axis=1)
    event = (obs > threshold).astype(float)
    return float(np.mean((prob - event) ** 2))


def interval_coverage(fc_ens, obs, nominal):
    alpha  = 1 - nominal
    lower  = np.quantile(fc_ens, alpha / 2,     axis=1)
    upper  = np.quantile(fc_ens, 1 - alpha / 2, axis=1)
    return {
        "nominal_coverage":   nominal,
        "empirical_coverage": float(np.mean((obs >= lower) & (obs <= upper))),
        "mean_width":         float(np.mean(upper - lower)),
    }


def rank_histogram(fc_ens, obs, rng=None):
    """Talagrand rank histogram with randomized ranks for ties.

    Precipitation has many exact zeros, so counting only members *strictly*
    below the observation assigns every tied case (e.g. a dry day where members
    are also 0) to rank 0, producing a spurious left spike. Following
    Hamill (2001) we draw the observation's rank uniformly within the tied
    block, so a perfectly dry ensemble contributes flat rather than to rank 0.
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


def reliability_diagram(fc_ens, obs, threshold):
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


def _crps_per_point(fc_ens, obs):
    term1 = np.abs(fc_ens - obs[:, None]).mean(axis=1)
    term2 = 0.5 * np.abs(fc_ens[:, :, None] - fc_ens[:, None, :]).mean(axis=(1, 2))
    return term1 - term2



# ── Legacy / alternative implementations (unused — kept for reference) ─────────
# The land-masked *field* formulations below predate the run_verification-derived
# canonical metrics above and are not imported anywhere. They are retained
# (commented out, not deleted) for provenance and possible future reuse.
#
# # ── try to delegate to scores library ─────────────────────────────────────────
#
# try:
#     from scores.probability import crps_for_ensemble as _scores_crps
#     _USE_SCORES = True
# except ImportError:
#     _USE_SCORES = False
#
#
# # ── Fair CRPS ─────────────────────────────────────────────────────────────────
#
# def fair_crps_field(
#     fc:  np.ndarray,   # (time, n_members, lat, lon)
#     obs: np.ndarray,   # (time, lat, lon)
# ) -> np.ndarray:
#     """
#     Per-grid-cell fair CRPS map  (time, lat, lon)  mm/day.
#
#     For deterministic forecasts (n_members == 1) returns MAE.
#     """
#     n = fc.shape[1]
#
#     if n == 1:
#         return np.abs(fc[:, 0] - obs)
#
#     # Score term: E|X - y|
#     score = np.abs(fc - obs[:, np.newaxis]).mean(axis=1)  # (t, lat, lon)
#
#     # Fair spread: S/(n*(n-1))  via sorted order-statistics formula
#     # sum_{i<j}|X_(j)-X_(i)| = sum_k (2k - n + 1) * X_(k)  [0-indexed]
#     fc_sorted = np.sort(fc, axis=1)                          # (t, n, lat, lon)
#     k  = np.arange(n, dtype=np.float64)
#     w  = (2 * k - n + 1).reshape(1, n, 1, 1)                # broadcast shape
#     spread = (fc_sorted * w).sum(axis=1) / (n * (n - 1))     # (t, lat, lon)
#
#     return (score - spread).astype(np.float32)
#
#
# def mean_fair_crps(
#     fc:   np.ndarray,
#     obs:  np.ndarray,
#     land: np.ndarray,
# ) -> float:
#     """Land-area mean fair CRPS (mm/day) — scalar summary for tables."""
#     crps_map = fair_crps_field(fc, obs)
#     return float(crps_map[:, land].mean())
#
#
# # ── Brier Score and BSS ────────────────────────────────────────────────────────
#
# def brier_score(
#     fc:            np.ndarray,   # (time, n_members, lat, lon)
#     obs:           np.ndarray,   # (time, lat, lon)
#     threshold_mm:  float,
#     land:          np.ndarray,   # (lat, lon)  bool
# ) -> dict:
#     """
#     Brier Score (BS) and Brier Skill Score (BSS) over land pixels.
#
#     BSS reference is the in-sample climatological hit rate (p_clim).
#     For a publishable BSS use the out-of-sample CHIRPS climatology instead
#     (pass the climatology model's exceedance probability as a 'clim_fc').
#     """
#     prob  = (fc > threshold_mm).mean(axis=1).astype(float)   # (t, lat, lon)
#     obs_b = (obs  > threshold_mm).astype(float)               # (t, lat, lon)
#
#     p_l   = prob [:, land].ravel()
#     o_l   = obs_b[:, land].ravel()
#
#     valid = np.isfinite(p_l) & np.isfinite(o_l)
#     p_l, o_l = p_l[valid], o_l[valid]
#
#     bs     = float(((p_l - o_l) ** 2).mean())
#     p_clim = float(o_l.mean())
#     bs_ref = p_clim * (1.0 - p_clim)
#     bss    = float(1.0 - bs / bs_ref) if bs_ref > 1e-6 else np.nan
#
#     return {"BS": bs, "BSS": bss, "p_clim": p_clim, "N": len(p_l)}
#
#
# # ── Rank histogram ─────────────────────────────────────────────────────────────
#
# def rank_histogram(
#     fc:   np.ndarray,   # (time, n_members, lat, lon)
#     obs:  np.ndarray,   # (time, lat, lon)
#     land: np.ndarray,   # (lat, lon)  bool
# ) -> dict:
#     """
#     Talagrand / rank histogram over all land pixels and times.
#
#     Interpretation of the shape:
#       flat     → well calibrated
#       U-shaped → under-dispersive (ensemble too narrow)
#       dome     → over-dispersive (ensemble too wide)
#       left skew → ensemble systematically too high
#       right skew → ensemble systematically too low
#
#     Returns
#     -------
#     dict:
#       counts   : (n_members + 1,) int — raw bin counts
#       rel_freq : (n_members + 1,) float — normalised relative frequency
#       delta    : float — normalised std dev (0 = perfectly flat)
#       n_total  : int
#     """
#     n_members = fc.shape[1]
#     n_bins    = n_members + 1
#
#     # Rank of obs among members (0 … n_members)
#     ranks = (fc < obs[:, np.newaxis]).sum(axis=1)  # (t, lat, lon)
#     vals  = ranks[:, land].ravel()
#     vals  = vals[~np.isnan(vals.astype(float))].astype(int)
#
#     counts, _ = np.histogram(vals, bins=np.arange(n_bins + 1) - 0.5)
#     expected  = len(vals) / n_bins
#     rel_freq  = counts / counts.sum() if counts.sum() > 0 else np.full(n_bins, np.nan)
#     delta     = float(np.std(counts) / expected) if expected > 0 else np.nan
#
#     return {
#         "counts":   counts,
#         "rel_freq": rel_freq,
#         "delta":    delta,
#         "n_total":  int(len(vals)),
#     }
#
#
# # ── Reliability diagram ────────────────────────────────────────────────────────
#
# def reliability(
#     fc:            np.ndarray,   # (time, n_members, lat, lon)
#     obs:           np.ndarray,   # (time, lat, lon)
#     land:          np.ndarray,   # (lat, lon)  bool
#     threshold_mm:  float,
#     n_bins:        int = 11,
# ) -> dict | None:
#     """
#     Murphy (1973) reliability decomposition.
#
#     Returns None if there are no observed events above the threshold.
#
#     Returns
#     -------
#     dict:
#       bin_centres : (n_bins,) float — forecast probability axis
#       obs_freq    : (n_bins,) float — observed relative frequency per bin
#       counts      : (n_bins,) int   — sample count per bin
#       clim        : float           — overall base rate
#       N           : int             — total sample count
#       BS          : float           — Brier Score
#       BSS         : float           — Brier Skill Score (vs clim reference)
#       REL         : float           — reliability component (lower = better)
#       RES         : float           — resolution component (higher = better)
#     """
#     prob  = (fc > threshold_mm).mean(axis=1).astype(float)  # (t, lat, lon)
#     obs_b = (obs > threshold_mm).astype(float)
#
#     p_l = prob [:, land].ravel()
#     o_l = obs_b[:, land].ravel()
#     valid = np.isfinite(p_l) & np.isfinite(o_l)
#     p_l, o_l = p_l[valid], o_l[valid]
#
#     if o_l.sum() == 0:
#         return None
#
#     edges   = np.linspace(0, 1, n_bins + 1)
#     centres = 0.5 * (edges[:-1] + edges[1:])
#     idx     = np.clip(np.digitize(p_l, edges) - 1, 0, n_bins - 1)
#
#     counts   = np.zeros(n_bins, dtype=int)
#     obs_freq = np.full(n_bins, np.nan)
#     for k in range(n_bins):
#         sel = idx == k
#         counts[k] = sel.sum()
#         if sel.sum() > 0:
#             obs_freq[k] = o_l[sel].mean()
#
#     clim   = float(o_l.mean())
#     bs     = float(((p_l - o_l) ** 2).mean())
#     bs_ref = clim * (1.0 - clim)
#     bss    = float(1.0 - bs / bs_ref) if bs_ref > 1e-6 else np.nan
#
#     # Murphy decomposition: BS = REL - RES + UNC
#     finite = np.isfinite(obs_freq)
#     n_tot  = len(p_l)
#     rel    = float(np.sum(counts[finite] * (centres[finite] - obs_freq[finite]) ** 2) / n_tot)
#     res    = float(np.sum(counts[finite] * (obs_freq[finite] - clim)            ** 2) / n_tot)
#
#     return {
#         "bin_centres": centres,
#         "obs_freq":    obs_freq,
#         "counts":      counts,
#         "clim":        clim,
#         "N":           n_tot,
#         "BS":          bs,
#         "BSS":         bss,
#         "REL":         rel,
#         "RES":         res,
#     }
#
#
# # ── Deterministic helpers ──────────────────────────────────────────────────────
#
# def det_metrics(
#     fc_mean: np.ndarray,   # (time, lat, lon)  ensemble mean or deterministic
#     obs:     np.ndarray,   # (time, lat, lon)
#     land:    np.ndarray,   # (lat, lon)  bool
# ) -> dict:
#     """RMSE, MAE, Bias in mm/day over all land pixels and times."""
#     f = fc_mean[:, land].ravel()
#     o = obs    [:, land].ravel()
#     valid = np.isfinite(f) & np.isfinite(o)
#     f, o  = f[valid], o[valid]
#     return {
#         "RMSE": float(np.sqrt(((f - o) ** 2).mean())),
#         "MAE":  float(np.abs(f - o).mean()),
#         "Bias": float((f - o).mean()),
#     }
#
