"""
Spread-Skill Ratio (SSR) vs lead day — GenCast only (the sole ensemble).

    SSR = spread / RMSE(ensemble mean)
    spread = sqrt( (M+1)/M * mean ensemble variance )      # Fortin et al. (2014)

A reliable ensemble has SSR = 1. SSR < 1 = under-dispersive (overconfident);
SSR > 1 = over-dispersive. FourCastNet / GraphCast are deterministic (no spread)
so they are excluded.

Two panels: (1) SSR vs lead day, (2) spread and RMSE separately, so the ratio is
interpretable (a change can come from spread or from error).

Usage (truth / climatology source defaults to CHIRPS):
    python ssr_lead_curves.py            # CHIRPS
    python ssr_lead_curves.py tamsat     # TAMSAT
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

import ea_common as C

TRUTH = (sys.argv[1] if len(sys.argv) > 1 else "chirps").lower()
LEADS = [1, 3, 5, 7]
MODEL = "gencast"

preds = C.load_predictions()
obs_2d, LAT, LON = C.load_truth(TRUTH)
M = preds[MODEL].sizes["sample"]
CORR = (M + 1) / M                       # finite-ensemble inflation


def spread_skill(lead):
    """Pooled (cells × cases) spread, RMSE and SSR at one lead day."""
    fc_all, ob_all = C.gather_pairs(preds, obs_2d, MODEL, lead)
    err = fc_all.mean(axis=1) - ob_all                 # ensemble-mean error
    var = fc_all.var(axis=1, ddof=1)                   # per-cell ensemble variance
    land = np.isfinite(ob_all)
    rmse   = np.sqrt(np.mean(err[land] ** 2))
    spread = np.sqrt(CORR * np.mean(var[land]))
    return spread, rmse, spread / rmse


spread, rmse, ssr = zip(*(spread_skill(ld) for ld in LEADS))
for ld, s, r, q in zip(LEADS, spread, rmse, ssr):
    print(f"LD{ld}  spread={s:.3f}  rmse={r:.3f}  SSR={q:.3f}")

# ── plot ──────────────────────────────────────────────────────────────────────
col = C.COLORS[MODEL]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# panel 1: SSR vs lead, with calibration target and dispersion regions
ax1.axhspan(0, 1, color="#fde4e1", alpha=0.6, zorder=0)          # under-dispersive
ax1.axhline(1.0, color="#444444", ls="--", lw=1.3, zorder=2,
            label="perfect calibration (SSR = 1)")
ax1.plot(LEADS, ssr, color=col, lw=2.6, marker="o", ms=9, mec="white", mew=1.0,
         zorder=3, label="GenCast")
ax1.text(LEADS[0], 0.04, "under-dispersive (overconfident)", fontsize=9,
         color="#b03a2e", va="bottom")
ax1.set_xticks(LEADS)
ax1.set_xlabel("Lead day", fontsize=12)
ax1.set_ylabel("Spread / RMSE", fontsize=12)
ax1.set_ylim(0, max(1.15, max(ssr) + 0.1))
ax1.set_title("Spread-skill ratio", fontsize=13, fontweight="bold")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=10, loc="upper right")

# panel 2: spread and RMSE separately
ax2.plot(LEADS, rmse,   color="#333333", lw=2.4, marker="s", ms=8, label="RMSE (ens. mean)")
ax2.plot(LEADS, spread, color=col,       lw=2.4, marker="o", ms=8, label="Spread (calibrated)")
ax2.set_xticks(LEADS)
ax2.set_xlabel("Lead day", fontsize=12)
ax2.set_ylabel("mm/day", fontsize=12)
ax2.set_ylim(0, max(rmse) * 1.15)
ax2.set_title("Ensemble spread vs error", fontsize=13, fontweight="bold")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=10, loc="best")

fig.suptitle(f"GenCast spread-skill, MAM 2024  (truth = {TRUTH.upper()}, M = {M})",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = f"mam2024_analysis_outputs/ssr_lead_curves_{TRUTH}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved →", out)
plt.show()
