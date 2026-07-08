"""
Event-based verification figures — plotted from the score tables.

Drawn from the tidy DataFrames that tables.py writes
(event_scores_by_threshold.csv, brier_scores.csv), so they can be produced
inside a run_verification run or regenerated standalone from the CSVs:

    python -m benchmark_ea.verification.event_plots <tables_dir> <out_dir>

Three figures per reference (CHIRPS, ERA5), all sharing one grammar —
one column per lead day, x = discrete event thresholds, one curve per model:

- event_skill_curves_<obs>: CSI (top) and frequency bias (bottom, FB = 1
  reference line).
- event_pod_far_<obs>: POD (top) and FAR (bottom) — the detection /
  false-alarm trade-off.
- brier_curves_<obs>: raw Brier score of the ensemble exceedance
  probabilities (ensemble models only), with the climatological base-rate
  Brier p(1 - p) as the dashed no-resolution reference.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmark_ea.verification.style import (
    FULL_WIDTH,
    INK2,
    MODEL_COLORS,
    MODEL_LABELS,
    REF_LINE,
    apply_style,
    grid_y,
    panel_label,
    savefig,
)


def _model_handles(models, lw=1.6):
    return [Line2D([0], [0], color=MODEL_COLORS[m], lw=lw,
                   label=MODEL_LABELS[m]) for m in models]


def _model_order(df):
    return [m for m in MODEL_COLORS if m in set(df["model"])]


def _threshold_axis(ax, x, thresholds):
    """Thresholds are discrete event definitions, not a continuous axis →
    equal spacing with '>thr' tick labels."""
    ax.set_xticks(x)
    ax.set_xticklabels([f">{thr:g}" for thr in thresholds])


def _event_pair_figure(df, models, lead_days, thresholds,
                       spec_top, spec_bot, suptitle, fname, out):
    """Two contingency metrics stacked (one row each), one column per lead
    day, one curve per model."""
    x = np.arange(len(thresholds))
    fig, axes = plt.subplots(2, len(lead_days), figsize=(FULL_WIDTH, 4.4),
                             sharey="row", squeeze=False)
    for c, ld in enumerate(lead_days):
        for r, spec in enumerate((spec_top, spec_bot)):
            ax = axes[r][c]
            for m in models:
                sub = (df[(df["model"] == m) & (df["lead_day"] == ld)]
                       .set_index("threshold_mm_day").loc[thresholds])
                ax.plot(x, sub[spec["key"]].to_numpy(),
                        color=MODEL_COLORS[m], lw=1.4, marker="o", ms=4,
                        mec="white", mew=0.6)
            if spec.get("ref") is not None:
                ax.axhline(spec["ref"], **REF_LINE)
            if r == 0:
                ax.set_title(f"Lead day {ld}")
            else:
                ax.set_xlabel("mm day$^{-1}$")
            _threshold_axis(ax, x, thresholds)
            grid_y(ax)
            panel_label(ax, "abcdefgh"[r * len(lead_days) + c])
    for r, spec in enumerate((spec_top, spec_bot)):
        axes[r][0].set_ylabel(spec["ylabel"])
        axes[r][0].set_ylim(*spec["ylim"])
        if spec.get("ref_text"):
            axes[r][-1].text(x[-1], spec["ref"], spec["ref_text"], fontsize=6,
                             color=INK2, va="bottom", ha="right")

    handles = _model_handles(models)
    for spec in (spec_top, spec_bot):
        if spec.get("ref_label"):
            handles.append(Line2D([0], [0], label=spec["ref_label"], **REF_LINE))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(suptitle)
    savefig(fig, out, fname)


def plot_event_scores(event_df, obs, out):
    """Event-based skill vs threshold from event_scores_by_threshold.csv, so
    figures and table always agree. Writes two figures:
    event_skill_curves_<obs> (CSI + frequency bias) and
    event_pod_far_<obs> (POD + FAR, the detection/false-alarm trade-off)."""
    df = event_df[event_df["obs"] == obs]
    lead_days = sorted(df["lead_day"].unique())
    thresholds = sorted(df["threshold_mm_day"].unique())
    models = _model_order(df)

    csi_max = df["csi"].max()
    fb_max = df["frequency_bias"].max()
    _event_pair_figure(
        df, models, lead_days, thresholds,
        {"key": "csi", "ylabel": "Critical Success Index (CSI)",
         "ylim": (0, max(0.6, 1.1 * csi_max))},
        {"key": "frequency_bias", "ylabel": "Frequency bias (FB)",
         "ylim": (0, max(1.35, 1.08 * fb_max)),
         "ref": 1.0, "ref_text": "unbiased ", "ref_label": "Unbiased (FB = 1)"},
        f"Event detection skill vs threshold, truth = {obs}, MAM 2024",
        f"event_skill_curves_{obs.lower()}", out)

    _event_pair_figure(
        df, models, lead_days, thresholds,
        {"key": "pod", "ylabel": "Probability of detection (POD)",
         "ylim": (0, 1)},
        {"key": "far", "ylabel": "False-alarm ratio (FAR)",
         "ylim": (0, 1)},
        "Event detection vs false alarms by threshold, "
        f"truth = {obs}, MAM 2024",
        f"event_pod_far_{obs.lower()}", out)


def plot_brier_curves(brier_df, obs, out):
    """Raw Brier score vs threshold, one column per lead day, ensemble models
    only (deterministic models have no forecast probabilities). The dashed
    curve is the climatological base-rate Brier p(1 - p) — the score of always
    forecasting the observed event frequency; below it = real resolution."""
    df = brier_df[brier_df["obs"] == obs]
    lead_days = sorted(df["lead_day"].unique())
    thresholds = sorted(df["threshold_mm_day"].unique())
    models = _model_order(df)
    x = np.arange(len(thresholds))

    fig, axes = plt.subplots(1, len(lead_days), figsize=(FULL_WIDTH, 2.6),
                             sharey=True, squeeze=False)
    for c, ld in enumerate(lead_days):
        ax = axes[0][c]
        for m in models:
            sub = (df[(df["model"] == m) & (df["lead_day"] == ld)]
                   .set_index("threshold_mm_day").loc[thresholds])
            ax.plot(x, sub["brier_score"].to_numpy(),
                    color=MODEL_COLORS[m], lw=1.4, marker="o", ms=4,
                    mec="white", mew=0.6)
        # base rate is model-independent; take it from any model's rows
        rate = (df[(df["model"] == models[0]) & (df["lead_day"] == ld)]
                .set_index("threshold_mm_day").loc[thresholds, "event_rate"]
                .to_numpy())
        ax.plot(x, rate * (1.0 - rate), **REF_LINE)
        ax.set_title(f"Lead day {ld}")
        ax.set_xlabel("mm day$^{-1}$")
        _threshold_axis(ax, x, thresholds)
        grid_y(ax)
        panel_label(ax, "abcd"[c])
    axes[0][0].set_ylabel("Brier score")
    axes[0][0].set_ylim(bottom=0)

    fig.legend(handles=_model_handles(models) + [
        Line2D([0], [0], label="Climatological base rate p(1 − p)", **REF_LINE),
    ], loc="outside lower center", ncol=len(models) + 1)
    fig.suptitle(f"Brier score by threshold, truth = {obs}, MAM 2024")
    savefig(fig, out, f"brier_curves_{obs.lower()}")


def plot_event_figures(event_df, brier_df, out, obs_labels=("CHIRPS", "ERA5")):
    """All event-based figures for run_verification: CSI+FB curves, POD+FAR
    curves and Brier-by-threshold curves per reference."""
    print("\n[11] Event-based figures …")
    for obs in obs_labels:
        plot_event_scores(event_df, obs, out)
        plot_brier_curves(brier_df, obs, out)


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: python -m benchmark_ea.verification.event_plots "
                 "<tables_dir> <out_dir>")
    tables_dir, out = argv
    os.makedirs(out, exist_ok=True)
    apply_style()
    event_df = pd.read_csv(os.path.join(tables_dir, "event_scores_by_threshold.csv"))
    brier_df = pd.read_csv(os.path.join(tables_dir, "brier_scores.csv"))
    plot_event_figures(event_df, brier_df, out)


if __name__ == "__main__":
    main(sys.argv[1:])
