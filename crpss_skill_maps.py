"""
2D CRPS skill score (CRPSS) maps vs the climatology baseline, lead day 1.

CRPSS = 1 - CRPS_model / CRPS_climatology   (per grid cell)
  > 0  model beats climatology   (skill)
  = 0  no better than climatology
  < 0  worse than climatology

The climatology baseline is the canonical out-of-sample CHIRPS day-of-year
ensemble (2000–2020, 21-member; see EXPERIMENTAL_SETUP.md), read from the
climatology-model predictions. CRPSS values come from the same function used by
``run_verification.py``, so they are identical to the benchmark's reported
numbers. Referenced to CHIRPS only.

Prerequisite — generate the climatology baseline once:
    ./run_inference.sh --models climatology --start 2024-01-01 --end 2024-12-24

Usage:
    python crpss_skill_maps.py [PRED_DIR]      # default: data/predictions
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
LEAD     = 1
PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/predictions"
START, END, OBS_END = "2024-03-01", "2024-05-31", "2024-06-10"
OUT = "mam2024_analysis_outputs/crpss_maps_ld1.png"

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

# CRPSS per model at lead day 1 — identical to run_verification's definition
crpss = V.crpss_maps_vs_climatology(preds, MODELS, obs_2d, INIT_DATES, LEAD)
for m in MODELS:
    valid = crpss[m][np.isfinite(crpss[m])]
    print(f"{m:12s}  CRPSS over wet cells  median={np.median(valid):+.3f}  "
          f"frac>0={np.mean(valid > 0):.2f}  (n={valid.size})")

# ── plot: CRPSS maps with coastlines, borders, lat gridlines + 0 contour ──────
proj = ccrs.PlateCarree()
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
extent = [LON.min() - 0.5, LON.max() + 0.5, LAT.min() - 0.5, LAT.max() + 0.5]
cmap = plt.get_cmap("RdBu").copy()
cmap.set_bad("#d9d9d9")

fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), subplot_kw={"projection": proj})
for ax, m in zip(axes, MODELS):
    sk_disp = np.clip(crpss[m], -1.0, 1.0)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN, facecolor="#dce9f5", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#f7f7f2", zorder=0)
    im = ax.pcolormesh(LON, LAT, np.ma.masked_invalid(sk_disp), norm=norm, cmap=cmap,
                       transform=proj, shading="nearest", zorder=1)
    ax.contour(LON, LAT, sk_disp, levels=[-0.5, -0.25, 0.25, 0.5],
               colors="#444444", linewidths=0.6, alpha=0.6, transform=proj, zorder=2)
    ax.contour(LON, LAT, sk_disp, levels=[0.0], colors="black",
               linewidths=1.8, transform=proj, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                      alpha=0.4, linestyle="--", zorder=5)
    gl.top_labels = gl.right_labels = False
    ax.set_title(V.MODEL_LABELS[m], fontsize=13, fontweight="bold")

cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                    pad=0.08, shrink=0.6)
cbar.set_label("CRPS skill score vs CHIRPS climatology   "
               "(blue = skill, red = worse than climatology, black line = 0)",
               fontsize=11)
fig.suptitle("2D CRPS skill score vs climatology, MAM 2024, lead day 1", fontsize=15,
             fontweight="bold", y=0.97)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved →", OUT)
plt.show()
