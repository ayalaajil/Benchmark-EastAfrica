#!/usr/bin/env python
"""MAM 2024 cumulative rainfall by East African country.

Emits one figure per country pair (``ea_cumulative_<A>_<B>.png``); each figure has
two side-by-side subplots, one country each. Every subplot shows cumulative
area-mean rainfall over the MAM 2024 season for every model (at the headline
forecast lead) against the CHIRPS / ERA5 / TAMSAT references. The flood window is
shaded. References keep their own colour+style (CHIRPS black solid, ERA5 green
dashed, TAMSAT brown dash-dot); models are solid, coloured per model.

How the cumulative rainfall is computed, per country and source:
  1. Daily fields are aligned to the valid day. References use the observed day;
     a model at lead L uses the forecast initialised L days earlier (init = valid-L).
     Complete, non-duplicated coverage of the 92-day season is asserted.
  2. Ensemble models (GenCast, Climatology) are collapsed to their member mean.
  3. Each day is reduced to one country value by a cos(lat)-area-weighted mean over
     the country's land cells (NaN cells dropped from numerator and weights alike).
  4. That daily series is cumulatively summed -> cumulative depth (mm) vs date.

Run from the repo root, e.g.::

    python mam2024_country_rainfall.py --pred-dir data/predictions \
        --out-dir mam2024_country_outputs
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmark_ea import analysis_io, domain
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.verification.data import load_observations

warnings.filterwarnings("ignore")

# --- fixed presentation choices (kept consistent with the flood notebook) -------
MODEL_LABELS = {"gencast": "GenCast", "graphcast": "GraphCast",
                "fourcastnet": "FourCastNet", "climatology": "Climatology",
                "neuralgcm": "NeuralGCM"}
MODEL_COLORS = {"gencast": "#2196F3", "graphcast": "#E53935",
                "fourcastnet": "#d4a017", "climatology": "#999999",
                "neuralgcm": "#6A1B9A"}
# References keep their own colour + line style.
REF_STYLE = {"CHIRPS": ("#000000", "-"), "ERA5": ("#2E7D32", "--"),
             "TAMSAT": ("#8B4513", "-.")}
REFS = ["CHIRPS", "ERA5", "TAMSAT"]

# Country pairings -- one figure per pair, two subplots each (Somalia alone).
COUNTRY_PAIRS = [("Kenya", "Tanzania"), ("Uganda", "Ethiopia"),
                 ("Rwanda", "Burundi"), ("Somalia",)]

FLOOD_WINDOW = ("2024-04-15", "2024-05-05")   # peak Kenya/Tanzania flooding


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_everything(pred_dir, data_dir, models, start, end):
    """Return (preds, refs, land, cmask, aw) for the season."""
    cfg = BenchmarkConfig(data_dir=data_dir)
    preds = analysis_io.load_predictions(pred_dir, models)
    chirps_da, era5_da, tamsat_da = load_observations(cfg, start, end, None)
    refs = {"CHIRPS": chirps_da, "ERA5": era5_da, "TAMSAT": tamsat_da}

    lat = preds[models[0]].lat.values
    lon = preds[models[0]].lon.values
    land = domain.land_mask(lat, lon).astype(bool)
    cmask = domain.country_masks(lat, lon)
    aw = domain.area_weights(lat, lon)
    return preds, refs, land, cmask, aw


# --------------------------------------------------------------------------- #
# Season alignment and regional aggregation
# --------------------------------------------------------------------------- #

def mam_daily(da, start, end, lead=None):
    """Valid-day-aligned daily field (valid_time, [sample,] lat, lon).

    Model: pass ``lead`` (init = valid - lead). Reference: ``lead=None``.
    Asserts complete, non-duplicated coverage of the season.
    """
    if lead is None:
        d = da.sel(time=slice(start, end)).rename({"time": "valid_time"})
    else:
        d = da.sel(lead_day=lead)
        vt = pd.to_datetime(d.init_time.values) + pd.Timedelta(days=int(lead))
        d = (d.assign_coords(valid_time=("init_time", vt))
               .swap_dims({"init_time": "valid_time"}).sortby("valid_time")
               .sel(valid_time=slice(start, end)))
    n_expected = pd.date_range(start, end).size
    assert d.valid_time.size == n_expected, (
        f"coverage hole: {d.valid_time.size} valid days, expected {n_expected} "
        f"(lead={lead})")
    assert pd.Index(pd.to_datetime(d.valid_time.values)).is_unique, "double-counted day"
    return d.compute()


def area_mean_series(daily, aw, land, cmask, country):
    """Cos(lat)-area-weighted country daily-mean series (pd.Series), NaN-safe.

    Ensemble members are averaged first. See the module docstring for the formula.
    """
    d = daily.mean("sample") if "sample" in daily.dims else daily
    w = aw * (land & cmask[country])
    ww = w.where(np.isfinite(d), 0.0)
    s = (d.where(np.isfinite(d), 0.0) * ww).sum(["lat", "lon"]) / ww.sum(["lat", "lon"])
    return pd.Series(s.values, index=pd.to_datetime(s.valid_time.values))


# --------------------------------------------------------------------------- #
# Figures -- one per country pair, two subplots (one country each)
# --------------------------------------------------------------------------- #
def _plot_country_axes(ax, country, D_model, D_ref, models, aw, land, cmask):
    """Draw one country's model-vs-reference cumulative curves on ``ax``."""
    for r in REFS:
        c, ls = REF_STYLE[r]
        s = area_mean_series(D_ref[r], aw, land, cmask, country).cumsum()
        ax.plot(s.index, s.values, color=c, ls=ls, lw=2.2, zorder=5)
    for m in models:
        s = area_mean_series(D_model[m], aw, land, cmask, country).cumsum()
        ax.plot(s.index, s.values, color=MODEL_COLORS[m], lw=1.4, alpha=0.9)
    ax.axvspan(pd.Timestamp(FLOOD_WINDOW[0]), pd.Timestamp(FLOOD_WINDOW[1]),
               color="#ffd6d6", alpha=0.5, zorder=0)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("cumulative rainfall (mm)")
    ax.set_title(f"{country} — cumulative MAM 2024 rainfall", fontsize=13,
                 fontweight="bold", color=domain.COUNTRIES.get(country, "#333333"))


def plot_country_pair(pair, D_model, D_ref, models, aw, land, cmask, lead, out):
    """One figure with a subplot per country in ``pair``. Returns the file path."""
    fig, axes = plt.subplots(1, len(pair), figsize=(9 * len(pair), 6), squeeze=False)
    for ax, country in zip(axes[0], pair):
        _plot_country_axes(ax, country, D_model, D_ref, models, aw, land, cmask)

    handles = ([Line2D([0], [0], color=REF_STYLE[r][0], ls=REF_STYLE[r][1], lw=2.2,
                       label=f"{r} (obs)") for r in REFS]
               + [Line2D([0], [0], color=MODEL_COLORS[m], lw=1.4, label=MODEL_LABELS[m])
                  for m in models]
               + [Patch(fc="#ffd6d6", alpha=0.5, label="flood window")])
    axes[0][0].legend(handles=handles, ncol=2, fontsize=9, loc="upper left")
    fig.suptitle(f"MAM 2024 cumulative rainfall (models at lead {lead} d vs references)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.autofmt_xdate()

    path = os.path.join(out, "ea_cumulative_" + "_".join(pair) + ".png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", default="data/predictions")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="mam2024_country_outputs")
    p.add_argument("--models", nargs="+",
                   default=["gencast", "graphcast", "fourcastnet", "neuralgcm",
                            "climatology"])
    p.add_argument("--lead", type=int, default=1, help="forecast lead (days)")
    p.add_argument("--start", default="2024-03-01")
    p.add_argument("--end", default="2024-05-31")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    preds, refs, land, cmask, aw = load_everything(
        args.pred_dir, args.data_dir, args.models, args.start, args.end)

    D_model = {m: mam_daily(preds[m], args.start, args.end, args.lead)
               for m in args.models}
    D_ref = {r: mam_daily(refs[r], args.start, args.end) for r in REFS}

    print("\nCHIRPS MAM 2024 season total (mm), area-weighted per country:")
    for name in domain.COUNTRIES:
        total = area_mean_series(D_ref["CHIRPS"], aw, land, cmask, name).sum()
        print(f"  {name:10s} {total:5.0f} mm")

    print("\nWrote:")
    for pair in COUNTRY_PAIRS:
        path = plot_country_pair(pair, D_model, D_ref, args.models, aw, land, cmask,
                                 args.lead, args.out_dir)
        print("  " + path)


if __name__ == "__main__":
    main()
