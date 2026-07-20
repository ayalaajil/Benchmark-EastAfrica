"""
RMSE-vs-lead-day and CRPS-vs-lead-day skill curves — plotted from the
per-season score tables (<out>/<season>/deterministic_skill_by_model_obs_lead.csv,
<out>/<season>/probabilistic_scores.csv), the same "read from the CSV so
figures and tables always agree" pattern used by event_plots.py, rather than
recomputing independently from the raw predictions.

    python -m benchmark_ea.verification.skill_lead_plots <out_dir>

Two plain per-metric figures per season folder, one column per truth source,
one curve per model — no composite/multi-metric panels, matching the style of
plot_acc_curves / plot_ssr_lead_curves in plots.py:

- rmse_lead_curves: RMSE of the ensemble-mean (or deterministic) forecast vs
  lead day, one panel per truth source (CHIRPS/ERA5/TAMSAT).
- crps_lead_curves: fair CRPS of the ensemble forecast vs lead day, one panel
  per truth source (CHIRPS/ERA5 — the two references the probabilistic table
  is computed against), ensemble models only.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from benchmark_ea.verification.seasons import SEASONS, season_title
from benchmark_ea.verification.style import (
    FULL_WIDTH,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
    grid_y,
    panel_label,
    savefig,
)


def _model_handles(models, lw=1.4):
    return [Line2D([0], [0], color=MODEL_COLORS[m], lw=lw, marker="o", ms=4,
                   mec="white", mew=0.6, label=MODEL_LABELS[m]) for m in models]


def _model_order(df):
    return [m for m in MODEL_COLORS if m in set(df["model"])]


def _lead_curve_figure(df, value_col, ylabel, obs_labels, title, fname, out):
    """One panel per truth source present in ``df``, x = lead day, one line
    per model, y = ``value_col``."""
    obs_present = [o for o in obs_labels if o in set(df["obs"])]
    if not obs_present:
        return
    lead_days = sorted(df["lead_day"].unique())
    models = _model_order(df)

    fig, axes = plt.subplots(1, len(obs_present), figsize=(FULL_WIDTH, 2.7),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, (ax, obs) in enumerate(zip(axes, obs_present)):
        sub_obs = df[df["obs"] == obs]
        for m in models:
            sub = (sub_obs[sub_obs["model"] == m]
                   .set_index("lead_day").reindex(lead_days))
            ax.plot(lead_days, sub[value_col].to_numpy(), color=MODEL_COLORS[m],
                    lw=1.4, marker="o", ms=4, mec="white", mew=0.6)
        ax.set_xticks(lead_days)
        ax.set_xlabel("Lead day")
        ax.set_title(f"vs {obs}")
        grid_y(ax)
        panel_label(ax, "abc"[i])
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)

    fig.legend(handles=_model_handles(models), loc="outside lower center",
              ncol=len(models))
    fig.suptitle(title)
    savefig(fig, out, fname)


def plot_rmse_vs_lead(df, out, season, obs_labels=("CHIRPS", "ERA5", "TAMSAT")):
    """RMSE vs lead day from one season's deterministic_skill_by_model_obs_lead.csv."""
    _lead_curve_figure(
        df, "rmse", "RMSE (mm day$^{-1}$)", obs_labels,
        f"RMSE vs lead day, {season_title(season)}",
        "rmse_lead_curves", out)


def plot_crps_vs_lead(df, out, season, obs_labels=("CHIRPS", "ERA5")):
    """Fair CRPS vs lead day from one season's probabilistic_scores.csv
    (ensemble models only)."""
    _lead_curve_figure(
        df, "mean_crps", "Fair CRPS (mm day$^{-1}$)", obs_labels,
        f"Ensemble CRPS vs lead day, {season_title(season)}",
        "crps_lead_curves", out)


def plot_skill_vs_lead_figures(out, seasons=SEASONS):
    """RMSE-vs-lead and CRPS-vs-lead curves, one set per season folder under
    ``out`` (each season's deterministic_skill_by_model_obs_lead.csv /
    probabilistic_scores.csv already lives in ``out/<season>/`` — see
    compute_and_save_tables). A season is skipped if its tables aren't
    present (e.g. a partial run)."""
    print("\n[12] Skill-vs-lead-day figures …")
    for season in seasons:
        season_dir = os.path.join(out, season)
        skill_path = os.path.join(season_dir, "deterministic_skill_by_model_obs_lead.csv")
        prob_path = os.path.join(season_dir, "probabilistic_scores.csv")
        if not (os.path.exists(skill_path) and os.path.exists(prob_path)):
            continue
        skill_df = pd.read_csv(skill_path)
        prob_df = pd.read_csv(prob_path)
        plot_rmse_vs_lead(skill_df, season_dir, season)
        plot_crps_vs_lead(prob_df, season_dir, season)


def main(argv):
    if len(argv) != 1:
        sys.exit("usage: python -m benchmark_ea.verification.skill_lead_plots <out_dir>\n"
                 "  <out_dir> must contain the per-season subfolders written by "
                 "compute_and_save_tables; figures are written into the same "
                 "per-season subfolders.")
    out, = argv
    apply_style()
    plot_skill_vs_lead_figures(out)


if __name__ == "__main__":
    main(sys.argv[1:])
