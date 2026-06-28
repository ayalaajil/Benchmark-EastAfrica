"""
Probabilistic and deterministic verification metrics.

"""

from __future__ import annotations

import numpy as np


# ── try to delegate to scores library ─────────────────────────────────────────

try:
    from scores.probability import crps_for_ensemble as _scores_crps
    _USE_SCORES = True
except ImportError:
    _USE_SCORES = False


# ── Fair CRPS ─────────────────────────────────────────────────────────────────

def fair_crps_field(
    fc:  np.ndarray,   # (time, n_members, lat, lon)
    obs: np.ndarray,   # (time, lat, lon)
) -> np.ndarray:
    """
    Per-grid-cell fair CRPS map  (time, lat, lon)  mm/day.

    For deterministic forecasts (n_members == 1) returns MAE.
    """
    n = fc.shape[1]

    if n == 1:
        return np.abs(fc[:, 0] - obs)

    # Score term: E|X - y|
    score = np.abs(fc - obs[:, np.newaxis]).mean(axis=1)  # (t, lat, lon)

    # Fair spread: S/(n*(n-1))  via sorted order-statistics formula
    # sum_{i<j}|X_(j)-X_(i)| = sum_k (2k - n + 1) * X_(k)  [0-indexed]
    fc_sorted = np.sort(fc, axis=1)                          # (t, n, lat, lon)
    k  = np.arange(n, dtype=np.float64)
    w  = (2 * k - n + 1).reshape(1, n, 1, 1)                # broadcast shape
    spread = (fc_sorted * w).sum(axis=1) / (n * (n - 1))     # (t, lat, lon)

    return (score - spread).astype(np.float32)


def mean_fair_crps(
    fc:   np.ndarray,
    obs:  np.ndarray,
    land: np.ndarray,
) -> float:
    """Land-area mean fair CRPS (mm/day) — scalar summary for tables."""
    crps_map = fair_crps_field(fc, obs)
    return float(crps_map[:, land].mean())


# ── Brier Score and BSS ────────────────────────────────────────────────────────

def brier_score(
    fc:            np.ndarray,   # (time, n_members, lat, lon)
    obs:           np.ndarray,   # (time, lat, lon)
    threshold_mm:  float,
    land:          np.ndarray,   # (lat, lon)  bool
) -> dict:
    """
    Brier Score (BS) and Brier Skill Score (BSS) over land pixels.

    BSS reference is the in-sample climatological hit rate (p_clim).
    For a publishable BSS use the out-of-sample CHIRPS climatology instead
    (pass the climatology model's exceedance probability as a 'clim_fc').
    """
    prob  = (fc > threshold_mm).mean(axis=1).astype(float)   # (t, lat, lon)
    obs_b = (obs  > threshold_mm).astype(float)               # (t, lat, lon)

    p_l   = prob [:, land].ravel()
    o_l   = obs_b[:, land].ravel()

    valid = np.isfinite(p_l) & np.isfinite(o_l)
    p_l, o_l = p_l[valid], o_l[valid]

    bs     = float(((p_l - o_l) ** 2).mean())
    p_clim = float(o_l.mean())
    bs_ref = p_clim * (1.0 - p_clim)
    bss    = float(1.0 - bs / bs_ref) if bs_ref > 1e-6 else np.nan

    return {"BS": bs, "BSS": bss, "p_clim": p_clim, "N": len(p_l)}


# ── Rank histogram ─────────────────────────────────────────────────────────────

def rank_histogram(
    fc:   np.ndarray,   # (time, n_members, lat, lon)
    obs:  np.ndarray,   # (time, lat, lon)
    land: np.ndarray,   # (lat, lon)  bool
) -> dict:
    """
    Talagrand / rank histogram over all land pixels and times.

    Interpretation of the shape:
      flat     → well calibrated
      U-shaped → under-dispersive (ensemble too narrow)
      dome     → over-dispersive (ensemble too wide)
      left skew → ensemble systematically too high
      right skew → ensemble systematically too low

    Returns
    -------
    dict:
      counts   : (n_members + 1,) int — raw bin counts
      rel_freq : (n_members + 1,) float — normalised relative frequency
      delta    : float — normalised std dev (0 = perfectly flat)
      n_total  : int
    """
    n_members = fc.shape[1]
    n_bins    = n_members + 1

    # Rank of obs among members (0 … n_members)
    ranks = (fc < obs[:, np.newaxis]).sum(axis=1)  # (t, lat, lon)
    vals  = ranks[:, land].ravel()
    vals  = vals[~np.isnan(vals.astype(float))].astype(int)

    counts, _ = np.histogram(vals, bins=np.arange(n_bins + 1) - 0.5)
    expected  = len(vals) / n_bins
    rel_freq  = counts / counts.sum() if counts.sum() > 0 else np.full(n_bins, np.nan)
    delta     = float(np.std(counts) / expected) if expected > 0 else np.nan

    return {
        "counts":   counts,
        "rel_freq": rel_freq,
        "delta":    delta,
        "n_total":  int(len(vals)),
    }


# ── Reliability diagram ────────────────────────────────────────────────────────

def reliability(
    fc:            np.ndarray,   # (time, n_members, lat, lon)
    obs:           np.ndarray,   # (time, lat, lon)
    land:          np.ndarray,   # (lat, lon)  bool
    threshold_mm:  float,
    n_bins:        int = 11,
) -> dict | None:
    """
    Murphy (1973) reliability decomposition.

    Returns None if there are no observed events above the threshold.

    Returns
    -------
    dict:
      bin_centres : (n_bins,) float — forecast probability axis
      obs_freq    : (n_bins,) float — observed relative frequency per bin
      counts      : (n_bins,) int   — sample count per bin
      clim        : float           — overall base rate
      N           : int             — total sample count
      BS          : float           — Brier Score
      BSS         : float           — Brier Skill Score (vs clim reference)
      REL         : float           — reliability component (lower = better)
      RES         : float           — resolution component (higher = better)
    """
    prob  = (fc > threshold_mm).mean(axis=1).astype(float)  # (t, lat, lon)
    obs_b = (obs > threshold_mm).astype(float)

    p_l = prob [:, land].ravel()
    o_l = obs_b[:, land].ravel()
    valid = np.isfinite(p_l) & np.isfinite(o_l)
    p_l, o_l = p_l[valid], o_l[valid]

    if o_l.sum() == 0:
        return None

    edges   = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.clip(np.digitize(p_l, edges) - 1, 0, n_bins - 1)

    counts   = np.zeros(n_bins, dtype=int)
    obs_freq = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = idx == k
        counts[k] = sel.sum()
        if sel.sum() > 0:
            obs_freq[k] = o_l[sel].mean()

    clim   = float(o_l.mean())
    bs     = float(((p_l - o_l) ** 2).mean())
    bs_ref = clim * (1.0 - clim)
    bss    = float(1.0 - bs / bs_ref) if bs_ref > 1e-6 else np.nan

    # Murphy decomposition: BS = REL - RES + UNC
    finite = np.isfinite(obs_freq)
    n_tot  = len(p_l)
    rel    = float(np.sum(counts[finite] * (centres[finite] - obs_freq[finite]) ** 2) / n_tot)
    res    = float(np.sum(counts[finite] * (obs_freq[finite] - clim)            ** 2) / n_tot)

    return {
        "bin_centres": centres,
        "obs_freq":    obs_freq,
        "counts":      counts,
        "clim":        clim,
        "N":           n_tot,
        "BS":          bs,
        "BSS":         bss,
        "REL":         rel,
        "RES":         res,
    }


# ── Deterministic helpers ──────────────────────────────────────────────────────

def det_metrics(
    fc_mean: np.ndarray,   # (time, lat, lon)  ensemble mean or deterministic
    obs:     np.ndarray,   # (time, lat, lon)
    land:    np.ndarray,   # (lat, lon)  bool
) -> dict:
    """RMSE, MAE, Bias in mm/day over all land pixels and times."""
    f = fc_mean[:, land].ravel()
    o = obs    [:, land].ravel()
    valid = np.isfinite(f) & np.isfinite(o)
    f, o  = f[valid], o[valid]
    return {
        "RMSE": float(np.sqrt(((f - o) ** 2).mean())),
        "MAE":  float(np.abs(f - o).mean()),
        "Bias": float((f - o).mean()),
    }
