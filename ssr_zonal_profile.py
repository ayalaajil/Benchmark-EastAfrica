"""
Zonal Spread-Skill Ratio (SSR) — GenCast, SSR by latitude across lead days.

For each latitude row we pool over longitude and cases:

    SSR(lat) = sqrt( (M+1)/M * <variance> ) / sqrt( <error^2> )

revealing *where* the ensemble is over/under-confident. SSR = 1 is calibrated;
< 1 under-dispersive (overconfident), > 1 over-dispersive. GenCast only (the
deterministic models have no spread).

Usage (truth / climatology source defaults to CHIRPS):
    python ssr_zonal_profile.py            # CHIRPS
    python ssr_zonal_profile.py tamsat     # TAMSAT
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

import ea_common as C

TRUTH = (sys.argv[1] if len(sys.argv) > 1 else "chirps").lower()
LEADS = [1, 3, 5, 7]
MODEL = "gencast"
ZONES = [(-12, -5, "Southern EA"), (-5, 5, "Equatorial belt"), (5, 15, "Semi-arid Horn")]

preds = C.load_predictions()
obs_2d, LAT, LON = C.load_truth(TRUTH)
M = preds[MODEL].sizes["sample"]
CORR = (M + 1) / M


def ssr_by_lat(lead):
    """Per-latitude SSR, averaging variance and error^2 over lon and cases."""
    fc_all, ob_all = C.gather_pairs(preds, obs_2d, MODEL, lead)
    err = fc_all.mean(axis=1) - ob_all
    var = fc_all.var(axis=1, ddof=1)
    land = np.isfinite(ob_all)
    err2 = np.where(land, err ** 2, np.nan)
    var  = np.where(land, var, np.nan)
    mean_err2 = np.nanmean(np.nanmean(err2, axis=0), axis=1)   # cases -> lon -> (lat,)
    mean_var  = np.nanmean(np.nanmean(var,  axis=0), axis=1)
    return np.sqrt(CORR * mean_var) / np.sqrt(mean_err2)


ssr = {ld: ssr_by_lat(ld) for ld in LEADS}
for ld in LEADS:
    print(f"LD{ld}  SSR(lat) min={np.nanmin(ssr[ld]):.2f}  "
          f"median={np.nanmedian(ssr[ld]):.2f}  max={np.nanmax(ssr[ld]):.2f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axvspan(-5, 5, color="#ffe7b3", alpha=0.45, lw=0, zorder=0)
for b in (-5, 5):
    ax.axvline(b, color="#cccccc", ls=":", lw=1, zorder=1)
ax.axhline(1.0, color="#444444", ls="--", lw=1.4, zorder=2,
           label="perfect calibration (SSR = 1)")

cmap = plt.get_cmap("viridis")
for i, ld in enumerate(LEADS):
    ax.plot(LAT, ssr[ld], color=cmap(i / (len(LEADS) - 1)), lw=2.2,
            marker="o", ms=4, mec="white", mew=0.5, zorder=3, label=f"Lead day {ld}")

ax.set_xlim(LAT.min(), LAT.max())
ax.set_xlabel("Latitude (°)    —    South  →  North", fontsize=12)
ax.set_ylabel("Spread / RMSE", fontsize=12)
ax.grid(alpha=0.3)

# headroom + climate-zone labels along the top
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax + 0.14 * (ymax - ymin))
for lo, hi, name in ZONES:
    ax.text((lo + hi) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9.5, color="#777777", fontweight="bold")

ax.legend(fontsize=10, loc="lower right", ncol=2, framealpha=0.95)
ax.set_title(f"GenCast zonal spread-skill ratio, MAM 2024  (truth = {TRUTH.upper()})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out = f"mam2024_analysis_outputs/ssr_zonal_{TRUTH}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved →", out)
plt.show()
