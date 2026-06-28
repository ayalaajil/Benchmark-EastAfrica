"""
2D CRPS skill score (CRPSS) maps vs the climatology baseline, for ALL lead days.

    CRPSS = 1 - CRPS_model / CRPS_climatology      (per grid cell)
      > 0 model beats climatology · < 0 worse · black contour = 0 boundary

Grid: rows = models, columns = lead days.

The climatology baseline is the canonical out-of-sample CHIRPS day-of-year
ensemble (2000–2020, 21-member; see EXPERIMENTAL_SETUP.md), read from the
climatology-model predictions. All CRPSS values come from the same functions
used by ``run_verification.py``, so the numbers here are identical to the
benchmark's reported CRPS skill score. CRPSS is referenced to CHIRPS only
(the product the climatology is built from).

Prerequisite — generate the climatology baseline once:
    ./run_inference.sh --models climatology --start 2024-01-01 --end 2024-12-24

Usage:
    python crpss_skill_maps_all_leads.py [PRED_DIR]     # default: data/predictions
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import warnings; warnings.filterwarnings("ignore")

import run_verification as V
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.truth import chirps as chirps_io

MODELS   = ["fourcastnet", "gencast", "graphcast"]
LEADS    = [1, 3, 5, 7]
PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/predictions"
START, END, OBS_END = "2024-03-01", "2024-05-31", "2024-06-07"
OUT = "mam2024_analysis_outputs/crpss_maps_all_leads_chirps.png"

# ── data + climatology baseline ───────────────────────────────────────────────
cfg   = BenchmarkConfig()
preds = V.load_predictions(PRED_DIR, MODELS)
clim  = V.load_climatology_reference(PRED_DIR)
if clim is None:
    sys.exit(f"No climatology predictions in {PRED_DIR}/climatology/. Generate them with:\n"
             "  ./run_inference.sh --models climatology --start 2024-01-01 --end 2024-12-24")
preds["climatology"] = clim

chirps_da = chirps_io.load(START, OBS_END, cfg.lat_vals, cfg.lon_vals,
                           cfg.chirps_cache_dir, download_missing=False)
obs_2d = {pd.Timestamp(t).date(): chirps_da.sel(time=t).values for t in chirps_da.time.values}
INIT_DATES = pd.date_range(START, END, freq="D")
LAT, LON = preds[MODELS[0]].lat.values, preds[MODELS[0]].lon.values

# CRPSS per (model, lead) — identical to run_verification's definition
SK = {}
for ld in LEADS:
    crpss_ld = V.crpss_maps_vs_climatology(preds, MODELS, obs_2d, INIT_DATES, ld)
    for m in MODELS:
        SK[(m, ld)] = crpss_ld[m]
for m in MODELS:
    line = "  ".join(f"LD{ld}:{np.nanmedian(SK[(m, ld)]):+.2f}" for ld in LEADS)
    print(f"{m:12s} median CRPSS  {line}")

# ── plot grid: rows = models, cols = lead days ────────────────────────────────
proj = ccrs.PlateCarree()
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
extent = [LON.min() - 0.5, LON.max() + 0.5, LAT.min() - 0.5, LAT.max() + 0.5]
cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#d9d9d9")

nrow, ncol = len(MODELS), len(LEADS)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow),
                         subplot_kw={"projection": proj})
for r, m in enumerate(MODELS):
    for c, ld in enumerate(LEADS):
        ax = axes[r, c]
        sk = np.clip(SK[(m, ld)], -1.0, 1.0)
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.OCEAN, facecolor="#dce9f5", zorder=0)
        ax.add_feature(cfeature.LAND,  facecolor="#f7f7f2", zorder=0)
        im = ax.pcolormesh(LON, LAT, np.ma.masked_invalid(sk), norm=norm, cmap=cmap,
                           transform=proj, shading="nearest", zorder=1)
        ax.contour(LON, LAT, sk, levels=[0.0], colors="black", linewidths=1.4,
                   transform=proj, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, linestyle=":", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.3, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (c == 0)
        gl.bottom_labels = (r == nrow - 1)
        if r == 0:
            ax.set_title(f"Lead day {ld}", fontsize=13, fontweight="bold")
        if c == 0:
            ax.text(-0.18, 0.5, V.MODEL_LABELS[m], transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=13, fontweight="bold")

cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04,
                    pad=0.04, shrink=0.5)
cbar.set_label("CRPS skill score vs CHIRPS climatology   "
               "(blue = skill, red = worse than climatology, black line = 0)",
               fontsize=11)
fig.suptitle("2D CRPS skill score vs climatology, MAM 2024, all lead days  (truth = CHIRPS)",
             fontsize=15, fontweight="bold", y=0.92)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved →", OUT)
plt.show()
