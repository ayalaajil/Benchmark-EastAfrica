"""
Per-country AI-model skill over East Africa (MAM 2024).

The companion to ``regional.py``: where that module characterises the *observed*
rainfall per country, this one asks **how well each AI model reproduces it**,
country by country. For every model × country × lead day it pools the matched
forecast/observation cells inside the country (cos-lat area-weighted) over the
MAM init dates and computes bias / MAE / RMSE (ensemble-mean) and, for ensemble
models, fair CRPS — reusing the same ``country_masks`` / ``area_weights`` and the
pure metric functions in ``benchmark_ea.metrics`` used everywhere else.

Runnable standalone:

    python -m benchmark_ea.verification.regional_skill \
        [--output-dir ./outputs_2024] [--pred-dir ./data/predictions] \
        [--data-dir ./data] [--year 2024] [--obs chirps]

Headline output is a set of spatial skill maps (a country average blends the wet
highlands with the arid lowlands into one meaningless number; the maps keep the
spatial structure). Outputs (into ``<output-dir>/regional/AI_skill_per_country/``):
    regional_best_model_<obs>.{pdf,png}    per-cell winner (lowest RMSE), by lead
    regional_bias_maps_<obs>.{pdf,png}     per-model wet/dry bias field (lead 1)
    regional_crpss_maps_<obs>.{pdf,png}    per-model skill vs climatology (lead 1)
    regional_disagreement_<obs>.{pdf,png}  inter-model spread, by lead
    regional_skill_by_country_<obs>.csv    per-country reference table (all leads)
"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from benchmark_ea import analysis_io
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.domain import (
    COUNTRIES,
    area_weights,
    country_masks,
    land_mask,
)
from benchmark_ea.metrics import crps_ensemble, deterministic_metrics
from benchmark_ea.verification.data import load_climatology_reference
from benchmark_ea.verification.plots import _map_chrome
from benchmark_ea.verification.scores import (
    crpss_maps_vs_climatology,
    spatial_metric_maps,
)
from benchmark_ea.verification.style import (
    CMAP_BIAS,
    CMAP_ERROR,
    CMAP_SKILL,
    FULL_WIDTH,
    INK,
    MODEL_COLORS,
    MODEL_LABELS,
    NAN_COLOR,
    apply_style,
    savefig,
)

OBS_LABELS = {"chirps": "CHIRPS", "tamsat": "TAMSAT"}
DEFAULT_MODELS = ["fourcastnet", "gencast", "graphcast", "neuralgcm"]


# ── Compute ───────────────────────────────────────────────────────────────────

def _country_pool(fc_all, ob_all, mask2d, w2d):
    """Pool a country's valid (finite-obs) cells across all cases.

    fc_all (case, sample, lat, lon), ob_all (case, lat, lon), mask2d/w2d
    (lat, lon). Returns (fc_mean_flat, ob_flat, w_flat, fc_ens) or None if the
    country has no valid cell/case."""
    sel = np.isfinite(ob_all) & mask2d[None]                  # (case, lat, lon)
    if not sel.any():
        return None
    fc_mean = fc_all.mean(axis=1)                             # (case, lat, lon)
    w_flat = np.broadcast_to(w2d[None], ob_all.shape)[sel]
    fc_ens = np.moveaxis(fc_all, 1, -1)[sel]                  # (n_valid, member)
    return fc_mean[sel], ob_all[sel], w_flat, fc_ens


def compute_skill_by_country(preds, models, obs_2d, init_dates, lead_days,
                             lat, lon, obs_label):
    """Per-country area-weighted skill for every model and lead day.

    Returns a DataFrame: country, model, lead_day, obs, bias, mae, rmse,
    pooled_corr, crps (NaN for deterministic models), n_pairs, n_cells."""
    masks = {c: m.values for c, m in country_masks(lat, lon).items()}
    w2d = area_weights(lat, lon).values

    rows = []
    for model in models:
        is_ens = preds[model].sizes.get("sample", 1) > 1
        for ld in lead_days:
            fc_all, ob_all = analysis_io.gather_pairs(preds, model, obs_2d,
                                                      init_dates, ld)
            if fc_all is None:
                continue
            for country in COUNTRIES:
                mask2d = masks.get(country)
                if mask2d is None or not mask2d.any():
                    continue
                pooled = _country_pool(fc_all, ob_all, mask2d, w2d)
                if pooled is None:
                    continue
                fc_flat, ob_flat, w_flat, fc_ens = pooled
                det = deterministic_metrics(fc_flat, ob_flat, w_flat)
                crps = np.nan
                if is_ens and fc_ens.shape[1] > 1:
                    crps = float(np.average(crps_ensemble(fc_ens, ob_flat),
                                            weights=w_flat))
                rows.append(dict(
                    country=country, model=model, lead_day=int(ld),
                    obs=obs_label, bias=det["bias"], mae=det["mae"],
                    rmse=det["rmse"], pooled_corr=det["pooled_corr"],
                    crps=crps, n_pairs=det["n"], n_cells=int(mask2d.sum()),
                ))
    return pd.DataFrame(rows)


# ── Spatial skill maps (keep the where, not just the how-much) ─────────────────
#
# Country averages blend the wet highlands with the arid lowlands into one
# meaningless number; these maps keep the spatial structure so a conclusion
# ("GenCast owns the coast, GraphCast the highlands") reads off directly.

def _extent(lat, lon):
    return [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]


def _rmse_stack(preds, models, obs_2d, init_dates, lead, land):
    """(n_models, lat, lon) land-masked RMSE at one lead, NaN where unavailable."""
    out = []
    for m in models:
        ds = spatial_metric_maps(preds, m, obs_2d, init_dates, lead)
        out.append(ds["rmse"].where(land).values if ds is not None
                   else np.full(land.shape, np.nan))
    return np.stack(out)


def plot_best_model_map(preds, models, obs_2d, init_dates, leads, lat, lon,
                        out, obs):
    """Per-cell winner: the model with the lowest ensemble-mean RMSE. One
    categorical panel per lead day, coloured by the model palette."""
    proj = ccrs.PlateCarree()
    extent = _extent(lat, lon)
    land = land_mask(lat, lon) > 0
    cmap = ListedColormap([MODEL_COLORS[m] for m in models])
    norm = BoundaryNorm(np.arange(-0.5, len(models) + 0.5, 1.0), cmap.N)

    fig, axes = plt.subplots(1, len(leads), figsize=(FULL_WIDTH, 2.9),
                             subplot_kw={"projection": proj}, squeeze=False)
    for c, ld in enumerate(leads):
        ax = axes[0][c]
        stack = _rmse_stack(preds, models, obs_2d, init_dates, ld, land)
        allnan = ~np.isfinite(stack).any(axis=0)
        winner = np.argmin(np.where(np.isfinite(stack), stack, np.inf),
                           axis=0).astype(float)
        winner[allnan] = np.nan
        _map_chrome(ax, extent, proj, left_labels=(c == 0), bottom_labels=True)
        ax.pcolormesh(lon, lat, np.ma.masked_invalid(winner), cmap=cmap,
                      norm=norm, transform=proj, shading="nearest", zorder=1)
        ax.set_title(f"Lead day {ld}")
    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="white",
                     label=MODEL_LABELS[m]) for m in models]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(models))
    fig.suptitle(f"Most skilful model per cell (lowest RMSE), "
                 f"MAM 2024 ({OBS_LABELS[obs]})")
    savefig(fig, out, f"regional_best_model_{obs}")


def _model_panels(lat, lon, models, title, fname, out, draw, cbar_label):
    """Shared 1×N (one panel per model) map scaffold; ``draw(ax, m)`` paints one
    model's field and returns the mappable (or None)."""
    proj = ccrs.PlateCarree()
    extent = _extent(lat, lon)
    fig, axes = plt.subplots(1, len(models), figsize=(FULL_WIDTH, 2.8),
                             subplot_kw={"projection": proj}, squeeze=False)
    im = None
    for c, m in enumerate(models):
        ax = axes[0][c]
        _map_chrome(ax, extent, proj, left_labels=(c == 0), bottom_labels=True)
        got = draw(ax, m, proj)
        im = got if got is not None else im
        ax.set_title(MODEL_LABELS[m])
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), orientation="horizontal",
                            fraction=0.05, pad=0.05, shrink=0.6)
        cbar.set_label(cbar_label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.4)
    fig.suptitle(title)
    savefig(fig, out, fname)


def plot_bias_maps(preds, models, obs_2d, init_dates, lead, lat, lon, out, obs):
    """Per-model wet/dry bias field (BrBG) at one lead — systematic spatial error."""
    land = land_mask(lat, lon) > 0
    ds_maps = {m: spatial_metric_maps(preds, m, obs_2d, init_dates, lead)
               for m in models}
    vals = np.concatenate([ds_maps[m]["bias"].where(land).values.ravel()
                           for m in models if ds_maps[m] is not None])
    vals = vals[np.isfinite(vals)]
    vmax = float(np.percentile(np.abs(vals), 97)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(CMAP_BIAS).copy()
    cmap.set_bad(NAN_COLOR)

    def draw(ax, m, proj):
        if ds_maps[m] is None:
            return None
        return ax.pcolormesh(
            lon, lat, np.ma.masked_invalid(ds_maps[m]["bias"].where(land).values),
            norm=norm, cmap=cmap, transform=proj, shading="nearest", zorder=1)

    _model_panels(
        lat, lon, models,
        f"Forecast bias by model, lead day {lead}, MAM 2024 ({OBS_LABELS[obs]})",
        f"regional_bias_maps_{obs}", out, draw,
        "Bias (mm day$^{-1}$) — brown too dry, green too wet")


def plot_crpss_maps(preds, models, obs_2d, init_dates, lead, lat, lon, out, obs):
    """Per-model CRPS skill score vs the climatology baseline (RdBu, 0-contour):
    where the AI actually beats 'just predict the seasonal normal'."""
    if "climatology" not in preds:
        print("  CRPSS maps: climatology predictions not loaded — skipped")
        return
    crpss = crpss_maps_vs_climatology(preds, models, obs_2d, init_dates, lead)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap(CMAP_SKILL).copy()
    cmap.set_bad(NAN_COLOR)

    def draw(ax, m, proj):
        sk = crpss.get(m)
        if sk is None:
            return None
        disp = np.clip(sk, -1.0, 1.0)
        im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(disp), norm=norm,
                           cmap=cmap, transform=proj, shading="nearest", zorder=1)
        ax.contour(lon, lat, np.where(np.isfinite(disp), disp, np.nan),
                   levels=[0.0], colors=INK, linewidths=0.7, transform=proj,
                   zorder=3)
        return im

    _model_panels(
        lat, lon, models,
        f"Skill vs climatology by model, lead day {lead}, MAM 2024 "
        f"({OBS_LABELS[obs]})",
        f"regional_crpss_maps_{obs}", out, draw,
        "CRPS skill score vs climatology (blue = beats climatology; "
        "black contour = 0; gray = hyper-arid)")


def plot_disagreement_map(preds, models, init_dates, leads, lat, lon, out, obs):
    """Where the models diverge most: time-mean spread across the four
    ensemble-mean forecasts. Observation-independent — flags low-confidence
    regions regardless of who is right."""
    proj = ccrs.PlateCarree()
    extent = _extent(lat, lon)
    land = land_mask(lat, lon) > 0

    fields = []
    for ld in leads:
        members = [preds[m].sel(lead_day=ld).mean("sample").sel(init_time=init_dates)
                   for m in models]
        spread = xr.concat(members, dim="model").std("model").mean("init_time")
        fields.append(spread.where(land))
    vmax = float(np.nanpercentile(
        np.concatenate([f.values.ravel() for f in fields]), 97))
    norm = plt.Normalize(vmin=0, vmax=vmax or 1.0)
    cmap = plt.get_cmap(CMAP_ERROR).copy()
    cmap.set_bad(NAN_COLOR)

    fig, axes = plt.subplots(1, len(leads), figsize=(FULL_WIDTH, 2.9),
                             subplot_kw={"projection": proj}, squeeze=False)
    im = None
    for c, (ld, field) in enumerate(zip(leads, fields)):
        ax = axes[0][c]
        _map_chrome(ax, extent, proj, left_labels=(c == 0), bottom_labels=True)
        im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field.values),
                           norm=norm, cmap=cmap, transform=proj,
                           shading="nearest", zorder=1)
        ax.set_title(f"Lead day {ld}")
    cbar = fig.colorbar(im, ax=list(axes[0]), orientation="horizontal",
                        fraction=0.05, pad=0.05, shrink=0.6)
    cbar.set_label("Inter-model spread (mm day$^{-1}$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)
    fig.suptitle(f"Where the models disagree most, MAM 2024 ({OBS_LABELS[obs]})")
    savefig(fig, out, f"regional_disagreement_{obs}")


def plot_skill_maps(preds, models, obs_2d, init_dates, leads, lat, lon, out, obs):
    """The four spatial skill maps."""
    primary = leads[0]
    print("  map: best model per cell …")
    plot_best_model_map(preds, models, obs_2d, init_dates, leads, lat, lon, out, obs)
    print(f"  map: bias by model (lead {primary}) …")
    plot_bias_maps(preds, models, obs_2d, init_dates, primary, lat, lon, out, obs)
    print(f"  map: skill vs climatology (lead {primary}) …")
    plot_crpss_maps(preds, models, obs_2d, init_dates, primary, lat, lon, out, obs)
    print("  map: model disagreement …")
    plot_disagreement_map(preds, models, init_dates, leads, lat, lon, out, obs)


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_skill_analysis(output_dir, data_dir, pred_dir, year, obs, models,
                       lead_days):
    cfg = BenchmarkConfig(data_dir=data_dir)
    out = os.path.join(output_dir, "regional/AI_skill_per_country")
    os.makedirs(out, exist_ok=True)
    label = OBS_LABELS[obs]

    print(f"\n=== Per-country skill + spatial skill maps, MAM {year} ({label}) ===")
    preds = analysis_io.load_predictions(pred_dir, models)
    lat = preds[models[0]].lat.values
    lon = preds[models[0]].lon.values

    # Climatology baseline (optional) → enables the CRPS-skill-vs-climatology map.
    clim = load_climatology_reference(pred_dir)
    if clim is not None:
        preds["climatology"] = clim
        print("  climatology baseline loaded (CRPSS map enabled)")
    else:
        print("  climatology baseline not found — CRPSS map will be skipped")

    start, end, obs_end = f"{year}-03-01", f"{year}-05-31", f"{year}-06-07"
    init_dates = pd.date_range(start, end, freq="D")
    obs_2d, _, _ = analysis_io.load_truth(
        obs, start, obs_end, lat, lon, data_dir,
        download_missing=(obs == "tamsat"))
    print(f"  {label}  {len(obs_2d)} daily grids")

    lead_days = [ld for ld in lead_days
                 if ld in [int(x) for x in preds[models[0]].lead_day.values]]

    # Per-country CSV kept as a quantitative reference table (the maps are the
    # headline output; a single national number hides the spatial structure).
    df = compute_skill_by_country(preds, models, obs_2d, init_dates, lead_days,
                                  lat, lon, label)
    if not df.empty:
        csv_path = os.path.join(out, f"regional_skill_by_country_{obs}.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  saved → {os.path.basename(csv_path)}")

    plot_skill_maps(preds, models, obs_2d, init_dates, lead_days, lat, lon,
                    out, obs)
    print(f"\nDone. Spatial skill maps + per-country table in {out}/")
    return df


def main(argv):
    p = argparse.ArgumentParser(
        description="Per-country AI-model skill over East Africa (MAM)")
    p.add_argument("--output-dir", default="./outputs_2024")
    p.add_argument("--data-dir", default="./data",
                   help="Root data dir; truth is read from <data-dir>/<obs>.")
    p.add_argument("--pred-dir", default="./data/predictions",
                   help="Dir with <model>/pred_YYYY-MM-DD.zarr.")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--obs", default="chirps", choices=list(OBS_LABELS),
                   help="Truth source to score against.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--lead-days", nargs="+", type=int, default=[1, 3, 5, 7])
    args = p.parse_args(argv)

    apply_style()
    run_skill_analysis(args.output_dir, args.data_dir, args.pred_dir,
                       args.year, args.obs, args.models, args.lead_days)


if __name__ == "__main__":
    main(sys.argv[1:])
