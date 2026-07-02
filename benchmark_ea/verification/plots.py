"""
Verification figures — all matplotlib/cartopy plotting for run_verification.

Isolates the heavy plotting stack. Scoring is done in benchmark_ea.verification.
scores; the pure metrics come from benchmark_ea.metrics.
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

from benchmark_ea.metrics import (
    rank_histogram,
    reliability_diagram_local,
    wilson_ci,
    compute_ece,
)
from benchmark_ea.verification.scores import (
    area_mean_ts,
    compute_temporal_metrics,
    crpss_maps_vs_climatology,
    gather_pairs,
    gather_pairs_with_local_thresh,
    spatial_metric_maps,
)

# ── Plot styling ──────────────────────────────────────────────────────────────

COLORS = {
    "fourcastnet": "#d4a017",
    "gencast":     "#2196F3",
    "graphcast":   "#E53935",
    "climatology": "#999999",
}
MODEL_LABELS = {
    "fourcastnet": "FourCastNet",
    "gencast":     "GenCast",
    "graphcast":   "GraphCast",
    "climatology": "Climatology",
}
MIN_COUNT_RELIABILITY = 200
PERCENTILES      = [20, 40, 60, 80]
PCTILE_LABELS    = {20: "Light rain", 40: "Moderate rain",
                    60: "Heavy rain",  80: "Intense rain"}


def savefig(fig, path, **kw):
    fig.savefig(path, dpi=150, bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  saved → {path}")


def _roll(series, win=7):
    return series.rolling(win, center=True, min_periods=3).mean()


def plot_crpss_maps(preds, models, obs_2d, init_dates, lead_days, out,
                    obs_label="chirps"):
    """CRPSS-vs-climatology maps, one row per lead day, one column per model."""
    from matplotlib.colors import TwoSlopeNorm

    leads = [ld for ld in lead_days]
    crpss_by_lead = {ld: crpss_maps_vs_climatology(preds, models, obs_2d,
                                                   init_dates, ld)
                     for ld in leads}
    lat = preds[models[0]].lat.values
    lon = preds[models[0]].lon.values
    proj = ccrs.PlateCarree()
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    extent = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#d9d9d9")

    nrow, ncol = len(leads), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.8 * nrow),
                             squeeze=False, subplot_kw={"projection": proj})
    im = None
    for r, ld in enumerate(leads):
        for c, m in enumerate(models):
            ax = axes[r][c]
            sk = crpss_by_lead[ld].get(m)
            ax.set_extent(extent, crs=proj)
            ax.add_feature(cfeature.OCEAN, facecolor="#dce9f5", zorder=0)
            ax.add_feature(cfeature.LAND,  facecolor="#f7f7f2", zorder=0)
            if sk is not None:
                disp = np.clip(sk, -1.0, 1.0)
                im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(disp),
                                   norm=norm, cmap=cmap, transform=proj,
                                   shading="nearest", zorder=1)
                ax.contour(lon, lat, disp, levels=[0.0], colors="black",
                           linewidths=1.6, transform=proj, zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=4)
            if r == 0:
                ax.set_title(MODEL_LABELS.get(m, m), fontsize=13, fontweight="bold")
            if c == 0:
                ax.text(-0.08, 0.5, f"lead {ld} d", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12,
                        fontweight="bold")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.05, shrink=0.5)
        cbar.set_label("CRPS skill score vs climatology  "
                       "(blue = skill, red = worse, black = 0)", fontsize=11)
    fig.suptitle(f"CRPS skill score vs climatology — {obs_label.upper()}",
                 fontsize=15, fontweight="bold", y=0.99)
    path = os.path.join(out, f"crpss_maps_vs_climatology_{obs_label}.png")
    savefig(fig, path)
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def plot_timeseries(preds, models, init_dates, lead_days,
                    chirps_lookup, era5_lookup, tamsat_lookup, out):
    print("\n[1] Timeseries …")
    ts_models = {m: {ld: area_mean_ts(preds, m, init_dates, ld) for ld in lead_days}
                 for m in models}

    fig, axes = plt.subplots(len(lead_days), 1,
                              figsize=(14, 4 * len(lead_days)), sharex=True)
    for ax, ld in zip(axes, lead_days):
        valid_dates = init_dates + pd.Timedelta(days=ld)
        for m in models:
            s = ts_models[m][ld]
            ax.plot(s.index, s.values, label=MODEL_LABELS[m],
                    color=COLORS[m], linewidth=1.4, alpha=0.85)
        ax.plot(valid_dates,
                [chirps_lookup.get(d.date(), np.nan) for d in valid_dates],
                "k-", lw=2, label="CHIRPS", zorder=5)
        ax.plot(valid_dates,
                [era5_lookup.get(d.date(), np.nan) for d in valid_dates],
                color="#2E7D32", ls="--", lw=2, label="ERA5", zorder=5)
        ax.plot(valid_dates,
                [tamsat_lookup.get(d.date(), np.nan) for d in valid_dates],
                color="#8B4513", ls="-.", lw=2, label="TAMSAT", zorder=5)
        ax.set_ylabel("mm / day", fontsize=10)
        ax.set_title(f"Lead day {ld}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)
        if ld == lead_days[0]:
            ax.legend(ncol=6, fontsize=9, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate()
    fig.suptitle("East Africa area-mean daily precipitation", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, os.path.join(out, "timeseries.png"))


def plot_temporal_skill(preds, models, init_dates, lead_days, chirps_2d, out):
    print("\n[2] Temporal skill curves …")
    temporal = {m: {ld: compute_temporal_metrics(preds, m, chirps_2d, init_dates, ld)
                    for ld in lead_days}
                for m in models}

    _obs_ts = pd.Series(
        {pd.Timestamp(d): float(np.nanmean(v)) for d, v in chirps_2d.items()}
    ).sort_index()

    nld = len(lead_days)
    _max_bias = max(
        temporal[m][ld]["bias"].abs().quantile(0.98)
        for m in models for ld in lead_days
    )

    # Bias + MAE
    fig, axes = plt.subplots(2, nld, figsize=(5.5 * nld, 10),
                              sharey="row", sharex="col")
    for col, ld in enumerate(lead_days):
        for m in models:
            axes[0, col].plot(_roll(temporal[m][ld]["bias"].dropna()),
                              color=COLORS[m], lw=2, label=MODEL_LABELS[m])
            axes[1, col].plot(_roll(temporal[m][ld]["mae"].dropna()),
                              color=COLORS[m], lw=2)
        axes[0, col].axhline(0, color="#333333", ls="--", lw=1, alpha=0.7)
        axes[0, col].set_ylim(-_max_bias, _max_bias)
        axes[0, col].set_title(f"Lead day {ld}", fontsize=12, fontweight="bold")
        for row in range(2):
            axes[row, col].grid(alpha=0.2, lw=0.5)
            axes[row, col].tick_params(labelsize=11)
            axes[row, col].xaxis.set_major_locator(mdates.MonthLocator())
            axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if col == 0:
            axes[0, col].set_ylabel("Bias (mm/day)", fontsize=14)
            axes[1, col].set_ylabel("MAE (mm/day)",  fontsize=14)

    fig.legend(handles=[Line2D([0], [0], color=COLORS[m], lw=2.5,
                               label=MODEL_LABELS[m]) for m in models],
               loc="lower center", ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Temporal Forecast Errors vs CHIRPS", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.08)
    savefig(fig, os.path.join(out, "temporal_skill_bias_mae.png"))

    # GenCast: CRPS / spread / SSR
    gc_metrics = ["crps", "spread", "spread_skill_ratio"]
    gc_labels  = ["CRPS (mm/day)", "Ensemble spread (mm/day)", "Spread / RMSE"]
    fig2, axes2 = plt.subplots(3, nld, figsize=(5.5 * nld, 10),
                                sharey="row", sharex="col")
    gc_color = COLORS["gencast"]
    for col, ld in enumerate(lead_days):
        df = temporal["gencast"][ld]
        for row, (metric, ylabel) in enumerate(zip(gc_metrics, gc_labels)):
            ax = axes2[row, col]
            ax.plot(_roll(df[metric].dropna()), color=gc_color, lw=2)
            ax.grid(alpha=0.2, lw=0.5)
            ax.tick_params(labelsize=11)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            if row == 0: ax.set_title(f"Lead day {ld}", fontsize=12, fontweight="bold")
            if col == 0: ax.set_ylabel(ylabel, fontsize=11)

    fig2.suptitle("GenCast: CRPS, spread, spread/skill ratio vs CHIRPS",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig2, os.path.join(out, "gencast_crps_spread.png"))

    return temporal


def plot_rank_histograms(preds, models, init_dates, lead_days,
                         chirps_2d, era5_2d, tamsat_2d, out):
    print("\n[3] Rank histograms …")
    obs_rows = [("CHIRPS", chirps_2d), ("ERA5", era5_2d), ("TAMSAT", tamsat_2d)]
    n_rows = len(obs_rows)

    fig, axes = plt.subplots(n_rows, len(lead_days),
                              figsize=(4 * len(lead_days), 4 * n_rows))
    for row, (obs_label, obs_2d) in enumerate(obs_rows):
        for col, ld in enumerate(lead_days):
            ax = axes[row, col]
            fc_ens, obs = gather_pairs(preds, "gencast", obs_2d, init_dates, ld)
            if obs.size == 0:
                ax.set_title(f"{obs_label} LD={ld}\n(no data)")
                continue
            freq   = rank_histogram(fc_ens, obs)
            n_bins = len(freq)
            ax.bar(range(n_bins), freq, color=COLORS["gencast"], alpha=0.75, width=0.8)
            ax.axhline(1.0 / n_bins, color="k", ls="--", lw=1.2)
            ax.set_title(f"Lead day {ld}", fontsize=10)
            ax.set_xlabel("Rank", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_ylim(0, 0.6)
            ax.grid(axis="y", alpha=0.3)
        axes[row, 0].annotate(
            obs_label, xy=(-0.32, 0.5), xycoords="axes fraction",
            fontsize=13, fontweight="bold", va="center", ha="center",
            rotation=90, annotation_clip=False,
        )

    fig.suptitle("GenCast rank histograms across lead days", fontsize=15)
    plt.tight_layout()
    savefig(fig, os.path.join(out, "rank_histograms.png"))


def plot_reliability_local(preds, init_dates, chirps_2d, era5_2d, tamsat_2d,
                            chirps_pctile, era5_pctile, tamsat_pctile, out):
    print("\n[4] Local-percentile reliability diagrams …")
    obs_setups = [
        ("CHIRPS", chirps_2d, chirps_pctile),
        ("ERA5",   era5_2d,   era5_pctile),
        ("TAMSAT", tamsat_2d, tamsat_pctile),
    ]
    n_rows = len(obs_setups)
    gc_color = COLORS["gencast"]

    # Pre-compute
    cache = {}
    for r, (lbl, obs_2d, tmaps) in enumerate(obs_setups):
        for c, q in enumerate(PERCENTILES):
            fc, ob, th = gather_pairs_with_local_thresh(
                preds, "gencast", obs_2d, tmaps[q], init_dates, lead_day=1)
            pl, of, ct = reliability_diagram_local(fc, ob, th)
            cache[(r, c)] = (np.array(pl), np.array(of), np.array(ct))

    fig, axes = plt.subplots(n_rows, len(PERCENTILES),
                              figsize=(4.5 * len(PERCENTILES), 4.5 * n_rows),
                              sharey=True, sharex=True)
    for row, (obs_label, _, _x) in enumerate(obs_setups):
        for col, q in enumerate(PERCENTILES):
            ax = axes[row, col]
            pl, of, ct = cache[(row, col)]
            lo_ci, hi_ci = wilson_ci(ct, of)
            ok = ct >= MIN_COUNT_RELIABILITY
            pv = pl[ok]

            ax.plot([0, 1], [0, 1], color="#333333", ls="--", lw=1, alpha=0.6, zorder=1)
            ax.axhline(q / 100, color="#999999", ls=":", lw=1.2, zorder=1)
            ax.fill_between([0, 1], [q / 100, q / 100], [0, 0],
                            color="#f5f5f5", zorder=0)
            if ok.any():
                ax.fill_between(pv, lo_ci[ok], hi_ci[ok],
                                color=gc_color, alpha=0.18, zorder=2, lw=0)
                ax.plot(pv, of[ok], "-", color=gc_color, lw=2, zorder=3)
                ax.scatter(pv, of[ok], s=55, color=gc_color,
                           edgecolors="white", linewidths=0.6, zorder=4)
                ece = compute_ece(pl[ok], of[ok], ct[ok])
                ax.text(0.97, 0.04, f"ECE={ece:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8.5, color="#333333",
                        bbox=dict(fc="white", ec="#cccccc", pad=2, lw=0.8))

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.2, lw=0.5, color="#aaaaaa")
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(f">P{q}  —  {PCTILE_LABELS[q]}", fontsize=11,
                             fontweight="bold", pad=6)
            if row == n_rows - 1:
                ax.set_xlabel("Forecast probability", fontsize=10)
            if col == 0:
                ax.set_ylabel("Observed frequency", fontsize=10)

        axes[row, 0].annotate(
            obs_label, xy=(-0.32, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", va="center", ha="center",
            rotation=90, annotation_clip=False,
        )

    fig.legend(handles=[
        Line2D([0], [0], color=gc_color, lw=2.5, label="GenCast"),
        Patch(facecolor=gc_color, alpha=0.3, label="95% Wilson CI"),
        Line2D([0], [0], color="#333333", ls="--", lw=1.2, label="Perfect reliability"),
        Line2D([0], [0], color="#999999", ls=":", lw=1.2, label="Climatology"),
    ], loc="lower center", ncol=4, fontsize=10,
       bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.95)

    fig.suptitle("GenCast reliability — local percentile thresholds (Lead day 1)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.subplots_adjust(left=0.10, bottom=0.07)
    savefig(fig, os.path.join(out, "reliability_local_percentile.png"))


def plot_spatial_maps(preds, models, init_dates, chirps_2d, era5_2d, lead_days, out):
    print("\n[5] Spatial bias/RMSE maps …")
    _CMAP  = {"bias": "RdBu_r", "mae": "YlOrRd", "rmse": "YlOrRd"}
    _LABEL = {"bias": "Bias (mm/day)", "mae": "MAE (mm/day)", "rmse": "RMSE (mm/day)"}
    proj   = ccrs.PlateCarree()
    lat    = preds[models[0]].lat.values
    lon    = preds[models[0]].lon.values
    extent = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]

    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
        for ld in [lead_days[0], lead_days[-1]]:
            ds_maps = {m: spatial_metric_maps(preds, m, obs_2d, init_dates, ld)
                       for m in models}
            metrics = ("bias", "rmse")
            fig, axes = plt.subplots(
                len(metrics), len(models),
                figsize=(5.8 * len(models), 4.2 * len(metrics)),
                subplot_kw={"projection": proj},
            )
            for row, metric in enumerate(metrics):
                all_vals = np.concatenate([
                    ds_maps[m][metric].values.flatten()
                    for m in models if ds_maps[m] is not None
                ])
                all_vals = all_vals[np.isfinite(all_vals)]
                if metric == "bias":
                    vmax = np.percentile(np.abs(all_vals), 97)
                    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                else:
                    norm = plt.Normalize(vmin=0, vmax=np.percentile(all_vals, 97))

                for col, m in enumerate(models):
                    ax = axes[row, col]
                    ax.set_extent(extent, crs=proj)
                    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
                    ax.add_feature(cfeature.OCEAN,     facecolor="#dce9f5", zorder=0)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
                    ax.add_feature(cfeature.BORDERS,   linewidth=0.45, linestyle=":", zorder=3)
                    if ds_maps[m] is not None:
                        im = ax.pcolormesh(lon, lat, ds_maps[m][metric].values,
                                           norm=norm, cmap=_CMAP[metric],
                                           transform=proj, shading="nearest", zorder=2)
                        fig.colorbar(im, ax=ax, orientation="vertical",
                                     shrink=0.85, pad=0.02).set_label(_LABEL[metric], fontsize=8)
                    if row == 0:
                        ax.set_title(MODEL_LABELS[m], fontsize=12, fontweight="bold", pad=6)
                    if col == 0:
                        ax.text(-0.18, 0.5, _LABEL[metric], va="center", ha="center",
                                rotation=90, transform=ax.transAxes, fontsize=10)

            fig.suptitle(f"Spatial error maps vs {obs_label} | lead day {ld}",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout(h_pad=1.0, w_pad=0.5)
            savefig(fig, os.path.join(out, f"spatial_maps_{obs_label.lower()}_ld{ld}.png"))
