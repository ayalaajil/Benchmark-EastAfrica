"""
Verification figures — all matplotlib/cartopy plotting for run_verification.

Isolates the heavy plotting stack. Scoring is done in benchmark_ea.verification.
scores; the pure metrics come from benchmark_ea.metrics; colors, fonts and the
PDF+PNG writer come from benchmark_ea.verification.style (single source of
truth for the publication style).

Figure conventions
------------------
- hue encodes exactly one thing per figure: model identity (categorical
  palette) or lead day (ordinal blue ramp) — never both;
- observations wear neutral inks with distinct dash patterns;
- dashed lines mean "reference/ideal", grids are solid hairlines;
- every figure is saved as vector PDF + 300-dpi PNG.
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmark_ea.domain import land_mask
from benchmark_ea.metrics import (
    rank_histogram,
    reliability_diagram_local,
    wilson_ci,
    compute_ece,
)
from benchmark_ea.verification.scores import (
    acc_pooled,
    area_mean_ts,
    crpss_maps_vs_climatology,
    gather_pairs,
    gather_pairs_with_local_thresh,
    seasonal_mean_field,
    spatial_metric_maps,
    spread_skill_pooled,
    ssr_by_lat,
)
from benchmark_ea.verification.seasons import filter_by_season, filter_index_by_season, season_title
from benchmark_ea.verification.style import (
    CMAP_BIAS,
    CMAP_ERROR,
    CMAP_SKILL,
    BORDER_LW,
    COAST_LW,
    FULL_WIDTH,
    GRID,
    INK,
    INK2,
    LAND_COLOR,
    MODEL_COLORS,
    MODEL_LABELS,
    MUTED,
    NAN_COLOR,
    OBS_STYLES,
    OCEAN_COLOR,
    REF_LINE,
    grid_y,
    lead_color,
    panel_label,
    savefig,
)

MIN_COUNT_RELIABILITY = 200
PERCENTILES   = [20, 40, 60, 80]
PCTILE_LABELS = {20: "Light rain", 40: "Moderate rain",
                 60: "Heavy rain",  80: "Intense rain"}


def _roll(series, win=7):
    return series.rolling(win, center=True, min_periods=3).mean()


def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def _lead_handles(lead_days):
    return [Line2D([0], [0], color=lead_color(i, len(lead_days)), lw=1.6,
                   label=f"Lead day {ld}") for i, ld in enumerate(lead_days)]


def _model_handles(models, lw=1.6):
    return [Line2D([0], [0], color=MODEL_COLORS[m], lw=lw,
                   label=MODEL_LABELS[m]) for m in models]


def _map_chrome(ax, extent, proj, left_labels=False, bottom_labels=False):
    """Base map styling: recessive land/ocean, thin coasts, hairline graticule."""
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=COAST_LW, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=BORDER_LW, zorder=4,
                   edgecolor=INK2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID,
                      alpha=0.8, linestyle="-")
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style = gl.ylabel_style = {"size": 5.5, "color": MUTED}
    return gl


# ── [1] Area-mean timeseries ──────────────────────────────────────────────────

def plot_timeseries(preds, models, init_dates, lead_days, season,
                    chirps_lookup, era5_lookup, tamsat_lookup, out):
    """Daily area-mean precipitation, one season at a time — a full-2024
    version is unreadably dense at daily resolution, so this is always
    scoped to one season (``season`` in benchmark_ea.verification.seasons.
    REAL_SEASONS); ``out`` is that season's output folder."""
    print(f"\n[1] Timeseries ({season}) …")
    season_dates = {ld: filter_by_season(init_dates, ld, season) for ld in lead_days}
    if all(len(season_dates[ld]) == 0 for ld in lead_days):
        print(f"  [skip timeseries — no {season} cases in this run]")
        return
    ts_models = {m: {ld: area_mean_ts(preds, m, season_dates[ld], ld) for ld in lead_days}
                 for m in models}
    obs_lookups = {"CHIRPS": chirps_lookup, "ERA5": era5_lookup,
                   "TAMSAT": tamsat_lookup}

    fig, axes = plt.subplots(len(lead_days), 1,
                             figsize=(FULL_WIDTH, 1.55 * len(lead_days) + 0.5),
                             sharex=True, sharey=True)
    for ax, ld in zip(np.atleast_1d(axes), lead_days):
        valid_dates = season_dates[ld] + pd.Timedelta(days=ld)
        for m in models:
            s = ts_models[m][ld]
            ax.plot(s.index, s.values, color=MODEL_COLORS[m], lw=1.1, alpha=0.9)
        for obs_label, lookup in obs_lookups.items():
            ax.plot(valid_dates,
                    [lookup.get(d.date(), np.nan) for d in valid_dates],
                    lw=1.3, zorder=5, **OBS_STYLES[obs_label])
        ax.set_ylabel("mm day$^{-1}$")
        ax.set_title(f"Lead day {ld}", loc="left")
        grid_y(ax)

    _month_axis(np.atleast_1d(axes)[-1])
    fig.legend(handles=_model_handles(models, lw=1.4) +
               [Line2D([0], [0], lw=1.4, label=k, **OBS_STYLES[k])
                for k in obs_lookups],
               loc="outside lower center", ncol=len(models) + 3)
    fig.suptitle(f"East Africa area-mean daily precipitation — {season_title(season)}")
    savefig(fig, out, "timeseries")


# ── [2] Temporal error curves ─────────────────────────────────────────────────

def plot_temporal_bias_mae(temporal, models, lead_days, season, out):
    """Bias and MAE vs valid date (7-day rolling mean), one column per lead,
    scoped to one season (see plot_timeseries) — ``temporal`` is the full
    annual dict from compute_temporal_metrics; this filters each series to
    ``season`` before plotting."""
    print(f"\n[2] Temporal bias/MAE curves ({season}) …")
    nld = len(lead_days)
    sub = {m: {ld: filter_index_by_season(temporal[m][ld], season) for ld in lead_days}
           for m in models}
    bias_maxes = [sub[m][ld]["bias"].abs().quantile(0.98) for m in models for ld in lead_days
                 if not sub[m][ld].empty]
    if not bias_maxes:
        print(f"  [skip temporal_bias_mae — no {season} cases in this run]")
        return
    max_bias = max(bias_maxes)

    fig, axes = plt.subplots(2, nld, figsize=(FULL_WIDTH, 3.6),
                             sharey="row", sharex="col")
    for col, ld in enumerate(lead_days):
        for m in models:
            axes[0, col].plot(_roll(sub[m][ld]["bias"].dropna()),
                              color=MODEL_COLORS[m], lw=1.2)
            axes[1, col].plot(_roll(sub[m][ld]["mae"].dropna()),
                              color=MODEL_COLORS[m], lw=1.2)
        axes[0, col].axhline(0, **REF_LINE)
        axes[0, col].set_ylim(-max_bias, max_bias)
        axes[0, col].set_title(f"Lead day {ld}")
        for row in range(2):
            grid_y(axes[row, col])
            _month_axis(axes[row, col])
    axes[0, 0].set_ylabel("Bias (mm day$^{-1}$)")
    axes[1, 0].set_ylabel("MAE (mm day$^{-1}$)")

    fig.legend(handles=_model_handles(models),
               loc="outside lower center", ncol=len(models))
    fig.suptitle(f"Forecast error vs CHIRPS (7-day rolling mean), {season_title(season)}")
    savefig(fig, out, "temporal_bias_mae")


def plot_ensemble_temporal(temporal, ens_models, lead_days, season, out):
    """CRPS, spread and spread/skill vs valid date; rows = ensemble models,
    columns = metrics, one curve per lead day (ordinal blue ramp), scoped to
    one season as in plot_temporal_bias_mae."""
    print(f"\n[3] Ensemble CRPS / spread / SSR curves ({season}) …")
    metrics = [("crps",               "CRPS (mm day$^{-1}$)"),
               ("spread",             "Spread (mm day$^{-1}$)"),
               ("spread_skill_ratio", "Spread / RMSE")]
    sub = {m: {ld: filter_index_by_season(temporal[m][ld], season) for ld in lead_days}
           for m in ens_models}
    if all(sub[m][ld].empty for m in ens_models for ld in lead_days):
        print(f"  [skip ensemble_crps_spread_ssr — no {season} cases in this run]")
        return

    nrow = len(ens_models)
    fig, axes = plt.subplots(nrow, 3, figsize=(FULL_WIDTH, 1.85 * nrow + 0.9),
                             sharex=True, sharey="col", squeeze=False)
    for r, m in enumerate(ens_models):
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            for i, ld in enumerate(lead_days):
                ax.plot(_roll(sub[m][ld][metric].dropna()),
                        color=lead_color(i, len(lead_days)), lw=1.2)
            grid_y(ax)
            _month_axis(ax)
            if r == 0:
                ax.set_title(label)
        axes[r, 0].set_ylabel(MODEL_LABELS[m], fontsize=8.5,
                              fontweight="bold", color=INK)

    fig.legend(handles=_lead_handles(lead_days),
               loc="outside lower center", ncol=len(lead_days))
    fig.suptitle(f"Ensemble skill and dispersion vs CHIRPS (7-day rolling mean), "
                f"{season_title(season)}")
    savefig(fig, out, "ensemble_crps_spread_ssr")


# ── [4] Rank histograms (tie-aware) ───────────────────────────────────────────

def plot_rank_histograms(preds, ens_models, init_dates, lead_days,
                         obs_sources, out):
    """Talagrand rank histograms with randomized tie ranks (Hamill 2001) —
    the appropriate form for rainfall, where exact zeros produce heavy ties.
    One figure per ensemble model; rows = observation datasets."""

    print("\n[4] Rank histograms (tie-aware) …")
    for model in ens_models:
        freqs = {}
        for obs_label, obs_2d in obs_sources.items():
            for ld in lead_days:
                fc_ens, obs = gather_pairs(preds, model, obs_2d, init_dates, ld)
                if obs.size:
                    freqs[(obs_label, ld)] = rank_histogram(fc_ens, obs)
        if not freqs:
            continue
        ymax = 1.15 * max(f.max() for f in freqs.values())

        n_rows, n_cols = len(obs_sources), len(lead_days)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(FULL_WIDTH, 1.45 * n_rows + 0.8),
                                 sharex=True, sharey=True, squeeze=False)
        for row, obs_label in enumerate(obs_sources):
            for col, ld in enumerate(lead_days):
                ax = axes[row, col]
                freq = freqs.get((obs_label, ld))
                if freq is None:
                    ax.set_axis_off()
                    continue
                n_bins = len(freq)
                ax.bar(range(n_bins), freq, color=MODEL_COLORS[model],
                       width=0.82)
                ax.axhline(1.0 / n_bins, **REF_LINE)
                ax.set_ylim(0, ymax)
                ax.set_xticks([0, (n_bins - 1) // 2, n_bins - 1])
                grid_y(ax)
                if row == 0:
                    ax.set_title(f"Lead day {ld}")
                if row == n_rows - 1:
                    ax.set_xlabel("Rank of observation")
            axes[row, 0].set_ylabel(f"{obs_label}\nfrequency")

        fig.legend(handles=[Line2D([0], [0], label="Uniform (calibrated)",
                                   **REF_LINE)],
                   loc="outside lower center")
        fig.suptitle(f"{MODEL_LABELS[model]} rank histograms 2024")
        savefig(fig, out, f"rank_histograms_{model}")


# ── [5] Reliability diagrams ──────────────────────────────────────────────────

def plot_reliability_local(preds, ens_models, init_dates, obs_setups, out):
    """Reliability at local percentile thresholds (lead day 1), one figure per
    ensemble model; rows = observation datasets, columns = percentiles."""
    print("\n[5] Local-percentile reliability diagrams …")
    n_rows = len(obs_setups)

    for model in ens_models:
        color = MODEL_COLORS[model]
        cache = {}
        for r, (lbl, obs_2d, tmaps) in enumerate(obs_setups):
            for c, q in enumerate(PERCENTILES):
                fc, ob, th = gather_pairs_with_local_thresh(
                    preds, model, obs_2d, tmaps[q], init_dates, lead_day=1)
                pl, of, ct = reliability_diagram_local(fc, ob, th)
                cache[(r, c)] = (np.array(pl), np.array(of), np.array(ct))

        fig, axes = plt.subplots(n_rows, len(PERCENTILES),
                                 figsize=(FULL_WIDTH, 1.72 * n_rows + 0.9),
                                 sharey=True, sharex=True, squeeze=False)
        for row, (obs_label, _o, _t) in enumerate(obs_setups):
            for col, q in enumerate(PERCENTILES):
                ax = axes[row, col]
                pl, of, ct = cache[(row, col)]
                lo_ci, hi_ci = wilson_ci(ct, of)
                ok = ct >= MIN_COUNT_RELIABILITY
                pv = pl[ok]

                ax.plot([0, 1], [0, 1], **REF_LINE)
                ax.axhline(q / 100, color=MUTED, ls=(0, (1, 1.2)), lw=0.8)
                if ok.any():
                    ax.fill_between(pv, lo_ci[ok], hi_ci[ok],
                                    color=color, alpha=0.15, lw=0)
                    ax.plot(pv, of[ok], color=color, lw=1.3)
                    ax.scatter(pv, of[ok], s=11, color=color, zorder=4,
                               edgecolors="white", linewidths=0.5)
                    ece = compute_ece(pl[ok], of[ok], ct[ok])
                    ax.text(0.96, 0.05, f"ECE {ece:.3f}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=6, color=INK2)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                if row == 0:
                    ax.set_title(f">P{q} · {PCTILE_LABELS[q]}")
                if row == n_rows - 1:
                    ax.set_xlabel("Forecast probability")
            axes[row, 0].set_ylabel(f"{obs_label}\nobserved frequency")

        fig.legend(handles=[
            Line2D([0], [0], color=color, lw=1.5, label=MODEL_LABELS[model]),
            Patch(facecolor=color, alpha=0.15, label="95% Wilson CI"),
            Line2D([0], [0], label="Perfect reliability", **REF_LINE),
            Line2D([0], [0], color=MUTED, ls=(0, (1, 1.2)), lw=0.8,
                   label="Climatological base rate"),
        ], loc="outside lower center", ncol=4)
        fig.suptitle(f"{MODEL_LABELS[model]} reliability, local percentile "
                     "thresholds, lead day 1")
        savefig(fig, out, f"reliability_{model}")


# ── [6] Spatial error maps ────────────────────────────────────────────────────

def plot_spatial_maps(preds, models, init_dates, chirps_2d, era5_2d,
                      lead_days, out):
    """Spatial bias/RMSE maps: one figure per (obs source, lead day), two
    rows (bias, RMSE), one column per model."""
    print("\n[6] Spatial bias/RMSE maps …")
    proj = ccrs.PlateCarree()
    lat = preds[models[0]].lat.values
    lon = preds[models[0]].lon.values
    extent = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    # Geography-based mask, not the obs product's own NaN pattern: CHIRPS is
    # NaN over ocean so those maps look right by accident, but ERA5 is a full
    # reanalysis with real values over open ocean and would otherwise paint
    # right over the OCEAN map feature.
    land = land_mask(lat, lon) > 0

    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
        for ld in lead_days:
            ds_maps = {m: spatial_metric_maps(preds, m, obs_2d, init_dates, ld)
                       for m in models}
            fig, axes = plt.subplots(
                2, len(models),
                figsize=(FULL_WIDTH, 4.1),
                subplot_kw={"projection": proj}, squeeze=False)

            for row, (metric, cmap, label) in enumerate([
                    ("bias", CMAP_BIAS,  "Bias (mm day$^{-1}$)"),
                    ("rmse", CMAP_ERROR, "RMSE (mm day$^{-1}$)")]):
                vals = np.concatenate([
                    ds_maps[m][metric].where(land).values.ravel()
                    for m in models if ds_maps[m] is not None])
                vals = vals[np.isfinite(vals)]
                if metric == "bias":
                    vmax = np.percentile(np.abs(vals), 97)
                    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                else:
                    norm = plt.Normalize(vmin=0, vmax=np.percentile(vals, 97))

                im = None
                for col, m in enumerate(models):
                    ax = axes[row, col]
                    _map_chrome(ax, extent, proj,
                                left_labels=(col == 0),
                                bottom_labels=(row == 1))
                    if ds_maps[m] is not None:
                        im = ax.pcolormesh(
                            lon, lat,
                            np.ma.masked_invalid(ds_maps[m][metric].where(land).values),
                            norm=norm, cmap=cmap, transform=proj,
                            shading="nearest", zorder=1)
                    if row == 0:
                        ax.set_title(MODEL_LABELS[m])
                if im is not None:
                    cbar = fig.colorbar(im, ax=list(axes[row]), shrink=0.85,
                                        pad=0.015, fraction=0.035)
                    cbar.set_label(label, fontsize=7)
                    cbar.ax.tick_params(labelsize=6)
                    cbar.outline.set_linewidth(0.4)

            fig.suptitle(f"Spatial forecast errors vs {obs_label}, "
                         f"lead day {ld}, 2024")
            savefig(fig, out, f"spatial_maps_{obs_label.lower()}_ld{ld}")


# ── [7] CRPSS maps vs climatology ─────────────────────────────────────────────

def plot_crpss_maps(preds, models, obs_2d, init_dates, lead_days, out,
                    obs_label="chirps"):
    """Per-cell CRPS skill score vs the out-of-sample climatology baseline;
    rows = models, columns = lead days. Hyper-arid cells are masked gray."""
    print("\n[7] CRPSS-vs-climatology maps …")
    crpss = {ld: crpss_maps_vs_climatology(preds, models, obs_2d,
                                           init_dates, ld)
             for ld in lead_days}
    lat = preds[models[0]].lat.values
    lon = preds[models[0]].lon.values
    proj = ccrs.PlateCarree()
    extent = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap(CMAP_SKILL).copy()
    cmap.set_bad(NAN_COLOR)

    nrow, ncol = len(models), len(lead_days)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(FULL_WIDTH, 1.62 * nrow + 1.2),
                             subplot_kw={"projection": proj}, squeeze=False)
    im = None
    for r, m in enumerate(models):
        for c, ld in enumerate(lead_days):
            ax = axes[r][c]
            _map_chrome(ax, extent, proj,
                        left_labels=(c == 0), bottom_labels=(r == nrow - 1))
            sk = crpss[ld].get(m)
            if sk is not None:
                disp = np.clip(sk, -1.0, 1.0)
                im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(disp),
                                   norm=norm, cmap=cmap, transform=proj,
                                   shading="nearest", zorder=1)
                ax.contour(lon, lat, disp, levels=[0.0], colors=INK,
                           linewidths=0.7, transform=proj, zorder=3)
            if r == 0:
                ax.set_title(f"Lead day {ld}")
            if c == 0:
                ax.text(-0.28, 0.5, MODEL_LABELS[m], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=8.5,
                        fontweight="bold", color=INK)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                            fraction=0.035, pad=0.045, shrink=0.55)
        cbar.set_label("CRPS skill score vs climatology "
                       "(blue = beats climatology; black contour = 0; "
                       "gray = hyper-arid, masked)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.4)
    fig.suptitle(f"CRPS skill score vs climatology, {obs_label.upper()}, "
                 "2024")
    savefig(fig, out, f"crpss_maps_{obs_label}")


# ── [8] ACC vs lead day ───────────────────────────────────────────────────────

def plot_acc_curves(preds, models, truth_sources, init_dates, lead_days, out):
    """Pooled anomaly correlation vs lead day, one panel per truth source.
    Anomalies are w.r.t. the truth's own per-cell seasonal mean."""
    print("\n[8] ACC lead curves …")
    acc = {}
    for t_label, obs_2d in truth_sources.items():
        clim_field = seasonal_mean_field(obs_2d)
        for m in models:
            acc[(t_label, m)] = [acc_pooled(preds, m, obs_2d, clim_field,
                                            init_dates, ld)
                                 for ld in lead_days]

    fig, axes = plt.subplots(1, len(truth_sources),
                             figsize=(FULL_WIDTH, 2.7), sharey=True)
    lo = min(min(v) for v in acc.values())
    for i, (ax, t_label) in enumerate(zip(np.atleast_1d(axes), truth_sources)):
        for m in models:
            ax.plot(lead_days, acc[(t_label, m)], color=MODEL_COLORS[m],
                    lw=1.4, marker="o", ms=4, mec="white", mew=0.6)
        ax.set_xticks(lead_days)
        ax.set_xlabel("Lead day")
        ax.set_title(f"vs {t_label}")
        ax.set_ylim(min(0.0, lo - 0.05), 1.0)
        grid_y(ax)
        panel_label(ax, "ab"[i])
    np.atleast_1d(axes)[0].set_ylabel("Anomaly correlation")

    fig.legend(handles=_model_handles(models),
               loc="outside lower center", ncol=len(models))
    fig.suptitle("Anomaly correlation coefficient vs lead day, 2024")
    savefig(fig, out, "acc_lead_curves")
    return acc


# ── [9] Spread-skill vs lead day ──────────────────────────────────────────────

def plot_ssr_lead_curves(preds, ens_models, obs_2d, init_dates, lead_days,
                         out, truth_label="chirps"):
    """(a) spread/skill ratio vs lead day; (b) its ingredients (spread, RMSE)
    so a change in the ratio can be attributed. Both ensembles."""
    print(f"\n[9] Spread-skill lead curves ({truth_label}) …")
    stats = {m: [spread_skill_pooled(preds, m, obs_2d, init_dates, ld)
                 for ld in lead_days] for m in ens_models}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7))
    for m in ens_models:
        spread, rmse, ssr = zip(*stats[m])
        ax1.plot(lead_days, ssr, color=MODEL_COLORS[m], lw=1.4,
                 marker="o", ms=4, mec="white", mew=0.6)
        ax2.plot(lead_days, rmse, color=MODEL_COLORS[m], lw=1.4,
                 marker="o", ms=4, mec="white", mew=0.6)
        ax2.plot(lead_days, spread, color=MODEL_COLORS[m], lw=1.4,
                 ls=(0, (4, 2)), marker="o", ms=4, mec="white", mew=0.6)

    ax1.axhline(1.0, **REF_LINE)
    ax1.text(lead_days[-1], 1.0, "calibrated ", fontsize=6, color=INK2,
             va="bottom", ha="right")
    ax1.set_ylim(0, max(1.1, 1.1 * max(s[2] for v in stats.values() for s in v)))
    ax1.set_ylabel("Spread / RMSE")
    ax2.set_ylabel("mm day$^{-1}$")
    for i, ax in enumerate((ax1, ax2)):
        ax.set_xticks(lead_days)
        ax.set_xlabel("Lead day")
        grid_y(ax)
        panel_label(ax, "ab"[i])

    fig.legend(handles=_model_handles(ens_models) + [
        Line2D([0], [0], color=INK2, lw=1.2, label="RMSE (ens. mean)"),
        Line2D([0], [0], color=INK2, lw=1.2, ls=(0, (4, 2)), label="Spread"),
    ], loc="outside lower center", ncol=4)
    fig.suptitle("Ensemble spread-skill vs lead day, "
                 f"truth = {truth_label.upper()}, 2024")
    savefig(fig, out, f"ssr_lead_curves_{truth_label}")


def plot_ssr_zonal(preds, ens_models, obs_2d, init_dates, lead_days, out,
                   truth_label="chirps"):
    """Zonal (per-latitude) spread/skill profile, one panel per ensemble;
    reveals where each ensemble is over/under-dispersive."""
    print(f"\n[10] Zonal spread-skill profiles ({truth_label}) …")
    lat = preds[ens_models[0]].lat.values

    fig, axes = plt.subplots(1, len(ens_models), figsize=(FULL_WIDTH, 2.7),
                             sharey=True, squeeze=False)
    for i, (ax, m) in enumerate(zip(axes[0], ens_models)):
        ax.axvspan(-5, 5, color="#f0efec", lw=0, zorder=0)
        for j, ld in enumerate(lead_days):
            ssr = ssr_by_lat(preds, m, obs_2d, init_dates, ld)
            if ssr is not None:
                ax.plot(lat, ssr, color=lead_color(j, len(lead_days)), lw=1.2)
        ax.axhline(1.0, **REF_LINE)
        ax.text(0, 0.02, "equatorial belt", fontsize=6, color=MUTED,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.set_xlim(lat.min(), lat.max())
        ax.set_xlabel("Latitude (°N)")
        ax.set_title(MODEL_LABELS[m])
        grid_y(ax)
        panel_label(ax, "ab"[i])
    axes[0][0].set_ylabel("Spread / RMSE")
    axes[0][0].set_ylim(bottom=0)

    fig.legend(handles=_lead_handles(lead_days) +
               [Line2D([0], [0], label="Calibrated (SSR = 1)", **REF_LINE)],
               loc="outside lower center", ncol=len(lead_days) + 1)
    fig.suptitle("Zonal spread-skill profile, "
                 f"truth = {truth_label.upper()}, 2024")
    savefig(fig, out, f"ssr_zonal_{truth_label}")
