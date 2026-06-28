"""
East Africa precipitation forecast verification.

Usage
-----
# MAM 2024 (default)
python run_verification.py

# Custom period
python run_verification.py --start 2024-01-01 --end 2024-12-31 \
                           --obs-end 2025-01-07 --output-dir ./outputs_2024

# Specific models only
python run_verification.py --models gencast graphcast
"""

import argparse
import glob
import os
import sys
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.truth import chirps as chirps_io
from benchmark_ea.truth import era5 as era5_io
from benchmark_ea.truth import tamsat as tamsat_io


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EA forecast verification")
    p.add_argument("--start",      default="2024-03-01")
    p.add_argument("--end",        default="2024-05-31")
    p.add_argument("--obs-end",    default="2024-06-07",
                   help="Last date needed for observations (init END + max lead day)")
    p.add_argument("--models",     nargs="+",
                   default=["fourcastnet", "gencast", "graphcast"],
                   help="Models to verify. 'climatology' is supported as a "
                        "baseline once its predictions have been generated.")
    p.add_argument("--lead-days",  nargs="+", type=int,
                   default=[1, 3, 5, 7])
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[1, 5, 10, 20],
                   help="mm/day thresholds for event-based scores")
    p.add_argument("--output-dir", default="./mam2024_analysis_outputs")
    p.add_argument("--pred-dir",   default="./data/predictions",
                   help="Dir containing <model>/pred_YYYY-MM-DD.zarr (the "
                        "benchmark_ea.run output dir). Only total_precipitation "
                        "is read, so precip-only and all-variable zarrs both work.")
    return p.parse_args()


# ── Constants ─────────────────────────────────────────────────────────────────

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
_Z95 = _norm.ppf(0.975)
MIN_COUNT_RELIABILITY = 200
PERCENTILES      = [20, 40, 60, 80]
PCTILE_LABELS    = {20: "Light rain", 40: "Moderate rain",
                    60: "Heavy rain",  80: "Intense rain"}
_PCTILE_KEYS     = [20, 40, 60, 75, 80]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(pred_dir, models):
    print("Loading model predictions …")
    preds = {}
    for m in models:
        files = sorted(glob.glob(f"{pred_dir}/{m}/pred_2024-*.zarr"))
        if not files:
            raise FileNotFoundError(f"No prediction files found for {m} in {pred_dir}/{m}/")
        parts = [xr.open_zarr(f)["total_precipitation"] for f in files]
        # Guard against mixing zarrs on different grids (e.g. stale global files
        # alongside freshly-regenerated East Africa ones), which would otherwise
        # silently misalign on concat.
        grids = {(p.sizes["lat"], p.sizes["lon"]) for p in parts}
        if len(grids) > 1:
            raise ValueError(
                f"{m}: prediction files have inconsistent lat/lon grids {grids} in "
                f"{pred_dir}/{m}/ — clear stale files and regenerate."
            )
        preds[m] = xr.concat(parts, dim="init_time")
        print(f"  {m:15s}  {dict(zip(preds[m].dims, preds[m].shape))}")
    return preds


def load_climatology_reference(pred_dir):
    """
    Load the climatology baseline predictions if present, for CRPS skill scores.

    Returns the total_precipitation DataArray (init_time, sample, lead_day,
    lat, lon) or None when the climatology predictions have not been generated.
    """
    files = sorted(glob.glob(f"{pred_dir}/climatology/pred_2024-*.zarr"))
    if not files:
        return None
    parts = [xr.open_zarr(f)["total_precipitation"] for f in files]
    grids = {(p.sizes["lat"], p.sizes["lon"]) for p in parts}
    if len(grids) > 1:
        raise ValueError(
            f"climatology: inconsistent lat/lon grids {grids} in "
            f"{pred_dir}/climatology/ — clear stale files and regenerate."
        )
    return xr.concat(parts, dim="init_time")


def load_observations(config, obs_end, output_dir):
    print("\nLoading observations …")
    start = "2024-03-01"
    chirps_da = chirps_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                               config.chirps_cache_dir, download_missing=False)
    print(f"  CHIRPS  {dict(zip(chirps_da.dims, chirps_da.shape))}")

    era5_da = era5_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                           config.data_dir + "/era5", download_missing=True)
    print(f"  ERA5    {dict(zip(era5_da.dims, era5_da.shape))}")

    tamsat_da = tamsat_io.load(start, obs_end, config.lat_vals, config.lon_vals,
                               config.data_dir + "/tamsat", download_missing=True)
    print(f"  TAMSAT  {dict(zip(tamsat_da.dims, tamsat_da.shape))}")

    return chirps_da, era5_da, tamsat_da


def build_lookup_dicts(chirps_da, era5_da, tamsat_da):
    chirps_2d = {pd.Timestamp(t).date(): chirps_da.sel(time=t).values
                 for t in chirps_da.time.values}
    era5_2d   = {pd.Timestamp(t).date(): era5_da.sel(time=t).values
                 for t in era5_da.time.values}
    tamsat_2d = {pd.Timestamp(t).date(): tamsat_da.sel(time=t).values
                 for t in tamsat_da.time.values}

    chirps_lookup  = {d: float(np.nanmean(v)) for d, v in chirps_2d.items()}
    era5_lookup    = {d: float(np.nanmean(v)) for d, v in era5_2d.items()}
    tamsat_lookup  = {d: float(np.nanmean(v)) for d, v in tamsat_2d.items()}

    return (chirps_2d, era5_2d, tamsat_2d,
            chirps_lookup, era5_lookup, tamsat_lookup)


# ── Core pair-gathering helpers ───────────────────────────────────────────────

def gather_pairs(preds, model, obs_2d, init_dates, lead_day=1):
    fc_da = preds[model].sel(lead_day=lead_day)
    fc_list, ob_list = [], []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values
        except Exception:
            continue
        ob = obs_2d[vd]
        fc_flat = fc.reshape(fc.shape[0], -1).T
        ob_flat = ob.flatten()
        mask = ~np.isnan(ob_flat)
        if not mask.any():
            continue
        fc_list.append(fc_flat[mask])
        ob_list.append(ob_flat[mask])
    return np.vstack(fc_list), np.concatenate(ob_list)


def gather_pairs_with_local_thresh(preds, model, obs_2d, thresh_map,
                                   init_dates, lead_day=1):
    fc_da = preds[model].sel(lead_day=lead_day)
    thresh_flat = thresh_map.flatten()
    fc_list, ob_list, th_list = [], [], []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values
        except Exception:
            continue
        ob = obs_2d[vd]
        fc_flat = fc.reshape(fc.shape[0], -1).T
        ob_flat = ob.flatten()
        mask = ~np.isnan(ob_flat) & ~np.isnan(thresh_flat)
        if not mask.any():
            continue
        fc_list.append(fc_flat[mask])
        ob_list.append(ob_flat[mask])
        th_list.append(thresh_flat[mask])
    return np.vstack(fc_list), np.concatenate(ob_list), np.concatenate(th_list)


# ── Metric functions ──────────────────────────────────────────────────────────

def area_mean_ts(preds, model, init_dates, lead_day):
    fc = preds[model].sel(lead_day=lead_day).mean("sample")
    vals = []
    for init in init_dates:
        try:
            v = float(fc.sel(init_time=init).mean(["lat", "lon"], skipna=True))
        except Exception:
            v = np.nan
        vals.append(v)
    return pd.Series(vals, index=init_dates + pd.Timedelta(days=lead_day))


def deterministic_metrics(fc_mean, obs):
    fc_mean = np.asarray(fc_mean, dtype=float)
    obs     = np.asarray(obs, dtype=float)
    mask    = np.isfinite(fc_mean) & np.isfinite(obs)
    fc_mean, obs = fc_mean[mask], obs[mask]
    err = fc_mean - obs
    out = {
        "n":        int(mask.sum()),
        "obs_mean": float(np.mean(obs)),
        "fc_mean":  float(np.mean(fc_mean)),
        "bias":     float(np.mean(err)),
        "mae":      float(np.mean(np.abs(err))),
        "rmse":     float(np.sqrt(np.mean(err ** 2))),
        "corr":     float(np.corrcoef(fc_mean, obs)[0, 1])
                    if len(obs) > 1 and np.std(fc_mean) > 0 and np.std(obs) > 0
                    else np.nan,
    }
    return out


def contingency_scores(fc_mean, obs, threshold):
    fc_e = np.asarray(fc_mean) > threshold
    ob_e = np.asarray(obs)     > threshold
    mask = np.isfinite(fc_mean) & np.isfinite(obs)
    fc_e, ob_e = fc_e[mask], ob_e[mask]
    hits  = np.sum( fc_e &  ob_e)
    miss  = np.sum(~fc_e &  ob_e)
    fa    = np.sum( fc_e & ~ob_e)
    cn    = np.sum(~fc_e & ~ob_e)
    return {
        "hits": int(hits), "misses": int(miss),
        "false_alarms": int(fa), "correct_negatives": int(cn),
        "pod":           hits / (hits + miss) if (hits + miss) > 0 else np.nan,
        "far":           fa   / (hits + fa)   if (hits + fa)   > 0 else np.nan,
        "csi":           hits / (hits + miss + fa) if (hits + miss + fa) > 0 else np.nan,
        "frequency_bias":(hits + fa) / (hits + miss) if (hits + miss) > 0 else np.nan,
        "observed_event_rate":  float(np.mean(ob_e)),
        "forecast_event_rate":  float(np.mean(fc_e)),
    }


def crps_ensemble(fc_ens, obs):
    term1 = np.mean(np.abs(fc_ens - obs[:, None]), axis=1)
    term2 = 0.5 * np.mean(np.abs(fc_ens[:, :, None] - fc_ens[:, None, :]), axis=(1, 2))
    return term1 - term2


def brier_score_ensemble(fc_ens, obs, threshold):
    prob  = (fc_ens > threshold).mean(axis=1)
    event = (obs > threshold).astype(float)
    return float(np.mean((prob - event) ** 2))


def interval_coverage(fc_ens, obs, nominal):
    alpha  = 1 - nominal
    lower  = np.quantile(fc_ens, alpha / 2,     axis=1)
    upper  = np.quantile(fc_ens, 1 - alpha / 2, axis=1)
    return {
        "nominal_coverage":   nominal,
        "empirical_coverage": float(np.mean((obs >= lower) & (obs <= upper))),
        "mean_width":         float(np.mean(upper - lower)),
    }


def rank_histogram(fc_ens, obs, rng=None):
    """Talagrand rank histogram with randomized ranks for ties.

    Precipitation has many exact zeros, so counting only members *strictly*
    below the observation assigns every tied case (e.g. a dry day where members
    are also 0) to rank 0, producing a spurious left spike. Following
    Hamill (2001) we draw the observation's rank uniformly within the tied
    block, so a perfectly dry ensemble contributes flat rather than to rank 0.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs     = np.asarray(obs)[:, None]
    n_below = (fc_ens < obs).sum(axis=1)                       # strictly below
    n_tied  = (fc_ens == obs).sum(axis=1)                      # exact ties
    # uniform integer offset in [0, n_tied] within the tied block
    ranks   = n_below + np.floor(rng.random(len(n_below)) * (n_tied + 1)).astype(int)
    n_bins  = fc_ens.shape[1] + 1
    counts  = np.bincount(ranks, minlength=n_bins)
    return counts / counts.sum()


def reliability_diagram(fc_ens, obs, threshold):
    n_members   = fc_ens.shape[1]
    prob_fc     = (fc_ens > threshold).sum(axis=1) / n_members
    bin_obs     = (obs > threshold).astype(float)
    prob_levels = np.arange(n_members + 1) / n_members
    obs_freq, counts = [], []
    for p in prob_levels:
        mask = np.isclose(prob_fc, p)
        counts.append(mask.sum())
        obs_freq.append(bin_obs[mask].mean() if mask.sum() > 0 else np.nan)
    return prob_levels, np.array(obs_freq), np.array(counts)


def reliability_diagram_local(fc_ens, obs_arr, thresh_arr, n_bins=10):
    n_members = fc_ens.shape[1]
    prob_fc   = (fc_ens > thresh_arr[:, None]).sum(axis=1) / n_members
    bin_obs   = (obs_arr > thresh_arr).astype(float)
    edges     = np.linspace(0, 1, n_bins + 1)
    prob_levels, obs_freq, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob_fc >= lo) & (prob_fc < hi) if lo > 0 else (prob_fc >= 0) & (prob_fc <= hi)
        ct   = int(mask.sum())
        counts.append(ct)
        prob_levels.append(float((lo + hi) / 2))
        obs_freq.append(float(bin_obs[mask].mean()) if ct > 0 else np.nan)
    return np.array(prob_levels), np.array(obs_freq), np.array(counts)


def wilson_ci(counts, obs_freq, z=_Z95):
    n = np.asarray(counts, dtype=float)
    p = np.asarray(obs_freq, dtype=float)
    lo, hi = np.full_like(p, np.nan), np.full_like(p, np.nan)
    ok = (n > 0) & ~np.isnan(p)
    nv, pv = n[ok], p[ok]
    denom  = 1 + z ** 2 / nv
    centre = (pv + z ** 2 / (2 * nv)) / denom
    margin = z * np.sqrt(pv * (1 - pv) / nv + z ** 2 / (4 * nv ** 2)) / denom
    lo[ok] = np.clip(centre - margin, 0, 1)
    hi[ok] = np.clip(centre + margin, 0, 1)
    return lo, hi


def compute_ece(prob_levels, obs_freq, counts):
    total = counts.sum()
    if total == 0:
        return np.nan
    return float(np.nansum(counts / total * np.abs(np.asarray(obs_freq) - np.asarray(prob_levels))))


def compute_pctile_maps(obs_2d):
    stack = np.stack(list(obs_2d.values()), axis=0)
    return {q: np.nanpercentile(stack, q, axis=0) for q in _PCTILE_KEYS}


def _crps_per_point(fc_ens, obs):
    term1 = np.abs(fc_ens - obs[:, None]).mean(axis=1)
    term2 = 0.5 * np.abs(fc_ens[:, :, None] - fc_ens[:, None, :]).mean(axis=(1, 2))
    return term1 - term2


def compute_temporal_metrics(preds, model, obs_2d, init_dates, lead_day):
    fc_da     = preds[model].sel(lead_day=lead_day)
    n_members = fc_da.sizes.get("sample", 1)
    is_ens    = n_members > 1
    records   = []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values
        except Exception:
            continue
        ob   = obs_2d[vd]
        mask = ~np.isnan(ob.flatten())
        if not mask.any():
            continue
        fc_flat = fc.reshape(n_members, -1)[:, mask]
        ob_flat = ob.flatten()[mask]
        fc_mean = fc_flat.mean(axis=0)
        bias = float(np.mean(fc_mean - ob_flat))
        mae  = float(np.mean(np.abs(fc_mean - ob_flat)))
        rmse = float(np.sqrt(np.mean((fc_mean - ob_flat) ** 2)))
        crps = spread = ssr = np.nan
        if is_ens:
            crps   = float(np.mean(_crps_per_point(fc_flat.T, ob_flat)))
            spread = float(np.mean(fc_flat.std(axis=0, ddof=1)))
            ssr    = spread / rmse if rmse > 0 else np.nan
        records.append(dict(
            valid_date=init + pd.Timedelta(days=lead_day),
            bias=bias, mae=mae, rmse=rmse,
            crps=crps, spread=spread, spread_skill_ratio=ssr,
        ))
    return pd.DataFrame(records).set_index("valid_date").sort_index()


def spatial_metric_maps(preds, model, obs_2d, init_dates, lead_day=1):
    fc_da = preds[model].sel(lead_day=lead_day)
    fc_means, obs_list = [], []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).mean("sample")
        except Exception:
            continue
        fc_means.append(fc)
        obs_list.append(xr.DataArray(
            obs_2d[vd], dims=("lat", "lon"),
            coords={"lat": fc_da.lat, "lon": fc_da.lon},
        ))
    if not fc_means:
        return None
    err = xr.concat(fc_means, dim="case") - xr.concat(obs_list, dim="case")
    return xr.Dataset({
        "bias": err.mean("case", skipna=True),
        "mae":  np.abs(err).mean("case", skipna=True),
        "rmse": np.sqrt((err ** 2).mean("case", skipna=True)),
    })


def gather_pairs_maps(preds, model, obs_2d, init_dates, lead_day):
    """Per-cell version of gather_pairs (keeps the lat/lon grid, no masking).

    Returns fc_all (case, sample, lat, lon) and ob_all (case, lat, lon); ocean
    cells stay NaN in ob_all so they drop out of the CRPS average.
    """
    fc_da = preds[model].sel(lead_day=lead_day)
    fc_list, ob_list = [], []
    for init in init_dates:
        vd = (init + pd.Timedelta(days=lead_day)).date()
        if vd not in obs_2d:
            continue
        try:
            fc = fc_da.sel(init_time=init).values   # (sample, lat, lon)
        except Exception:
            continue
        fc_list.append(fc)
        ob_list.append(obs_2d[vd])
    if not fc_list:
        return None, None
    return np.stack(fc_list), np.stack(ob_list)


def _crps_map(fc_all, ob_all):
    """Mean fair CRPS per grid cell. fc_all (case, sample, lat, lon)."""
    term1 = np.abs(fc_all - ob_all[:, None]).mean(axis=1)                 # (case, lat, lon)
    term2 = 0.5 * np.abs(fc_all[:, :, None] - fc_all[:, None, :]).mean(axis=(1, 2))
    return np.nanmean(term1 - term2, axis=0)                              # (lat, lon)


def crpss_maps_vs_climatology(preds, models, obs_2d, init_dates,
                              lead_day=1, arid_thresh=0.5):
    """
    Per-cell CRPS skill score of each model vs the climatology baseline:

        CRPSS = 1 - CRPS_model / CRPS_climatology      (both on the same days)

    The climatology CRPS is computed from the climatology *model* predictions
    (out-of-sample 21-year ensemble), scored on the identical valid dates. Cells
    where mean observed rainfall < ``arid_thresh`` mm/day are masked (near-zero
    denominator → uninformative). Returns {model: (lat, lon) array}.
    """
    crps_clim = _crps_map(*gather_pairs_maps(preds, "climatology", obs_2d,
                                             init_dates, lead_day))
    # mask hyper-arid cells via the observed-mean field over the matched days
    _, ob_all = gather_pairs_maps(preds, models[0], obs_2d, init_dates, lead_day)
    arid = np.nanmean(ob_all, axis=0) < arid_thresh

    out = {}
    for m in models:
        fc_all, ob_m = gather_pairs_maps(preds, m, obs_2d, init_dates, lead_day)
        if fc_all is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sk = 1.0 - _crps_map(fc_all, ob_m) / crps_clim
        out[m] = np.where(arid | ~np.isfinite(sk), np.nan, sk)
    return out


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


def _roll(series, win=7):
    return series.rolling(win, center=True, min_periods=3).mean()


def savefig(fig, path, **kw):
    fig.savefig(path, dpi=150, bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  saved → {path}")


# ── Section 1: Timeseries ─────────────────────────────────────────────────────

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


# ── Section 2: Temporal skill curves ─────────────────────────────────────────

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


# ── Section 3: Rank histograms ────────────────────────────────────────────────

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


# ── Section 4: Local-percentile reliability diagrams ─────────────────────────

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


# ── Section 5: Spatial maps ───────────────────────────────────────────────────

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


# ── Section 6: CSV outputs ────────────────────────────────────────────────────

def compute_and_save_tables(preds, models, init_dates, lead_days_analysis,
                             chirps_2d, era5_2d, tamsat_2d,
                             pairs_chirps, pairs_era5, pairs_tamsat,
                             thresholds, out):
    print("\n[6] Computing and saving CSV tables …")

    # 6a. Summary bias table (lead day 1)
    rows = []
    for m in models:
        for obs_label, pairs in [("CHIRPS", pairs_chirps),
                                  ("ERA5",   pairs_era5),
                                  ("TAMSAT", pairs_tamsat)]:
            fc_ens, obs = pairs[m]
            fc_mean = fc_ens.mean(axis=1)
            rows.append({
                "model": m, "vs": obs_label,
                "bias (mm/d)":  round(float(np.nanmean(fc_mean - obs)),            3),
                "MAE (mm/d)":   round(float(np.nanmean(np.abs(fc_mean - obs))),    3),
                "RMSE (mm/d)":  round(float(np.sqrt(np.nanmean((fc_mean-obs)**2))), 3),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary_bias_table.csv"), index=False)
    print("  summary_bias_table.csv")

    # 6b. Deterministic skill across all lead days
    skill_rows = []
    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d), ("TAMSAT", tamsat_2d)]:
        for ld in lead_days_analysis:
            for m in models:
                fc_ens, obs = gather_pairs(preds, m, obs_2d, init_dates, ld)
                row = deterministic_metrics(fc_ens.mean(axis=1), obs)
                row.update({"model": m, "obs": obs_label, "lead_day": ld})
                skill_rows.append(row)
    skill_df = pd.DataFrame(skill_rows)
    skill_df.to_csv(os.path.join(out, "deterministic_skill_by_model_obs_lead.csv"), index=False)
    print("  deterministic_skill_by_model_obs_lead.csv")

    # 6c. Event-based scores
    event_rows = []
    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
        for ld in lead_days_analysis:
            for m in models:
                fc_ens, obs = gather_pairs(preds, m, obs_2d, init_dates, ld)
                fc_mean = fc_ens.mean(axis=1)
                for thr in thresholds:
                    row = contingency_scores(fc_mean, obs, thr)
                    row.update({"model": m, "obs": obs_label,
                                "lead_day": ld, "threshold_mm_day": thr})
                    event_rows.append(row)
    pd.DataFrame(event_rows).to_csv(os.path.join(out, "event_scores_by_threshold.csv"), index=False)
    print("  event_scores_by_threshold.csv")

    # 6d. GenCast probabilistic scores (CRPS, Brier, interval coverage)
    prob_rows, cov_rows, brier_rows = [], [], []
    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
        for ld in lead_days_analysis:
            fc_ens, obs = gather_pairs(preds, "gencast", obs_2d, init_dates, ld)
            crps    = crps_ensemble(fc_ens, obs)
            fc_mean = fc_ens.mean(axis=1)
            spread  = fc_ens.std(axis=1, ddof=1)
            rmse    = float(np.sqrt(np.mean((fc_mean - obs) ** 2)))
            prob_rows.append({
                "model": "gencast", "obs": obs_label, "lead_day": ld,
                "n":                    len(obs),
                "mean_crps":            float(np.mean(crps)),
                "rmse_ensemble_mean":   rmse,
                "mean_ensemble_spread": float(np.mean(spread)),
                "spread_skill_ratio":   float(np.mean(spread)) / rmse if rmse > 0 else np.nan,
            })
            for nominal in [0.50, 0.80, 0.90]:
                row = interval_coverage(fc_ens, obs, nominal)
                row.update({"model": "gencast", "obs": obs_label, "lead_day": ld})
                cov_rows.append(row)
            for thr in thresholds:
                brier_rows.append({
                    "model": "gencast", "obs": obs_label,
                    "lead_day": ld, "threshold_mm_day": thr,
                    "brier_score": brier_score_ensemble(fc_ens, obs, thr),
                    "event_rate":  float(np.mean(obs > thr)),
                })
    pd.DataFrame(prob_rows).to_csv(os.path.join(out, "gencast_probabilistic_scores.csv"), index=False)
    pd.DataFrame(cov_rows).to_csv(os.path.join(out, "gencast_interval_coverage.csv"),     index=False)
    pd.DataFrame(brier_rows).to_csv(os.path.join(out, "gencast_brier_scores.csv"),        index=False)
    print("  gencast_probabilistic_scores.csv / interval_coverage.csv / brier_scores.csv")

    # 6e. GenCast reliability tables + ECE
    rel_rows = []
    for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
        for ld in lead_days_analysis:
            fc_ens, obs = gather_pairs(preds, "gencast", obs_2d, init_dates, ld)
            for thr in thresholds:
                pl, of, ct = reliability_diagram(fc_ens, obs, thr)
                weights    = ct / ct.sum() if ct.sum() > 0 else np.zeros_like(ct, float)
                abs_gap    = np.abs(of - pl)
                ece        = float(np.nansum(weights * abs_gap))
                for i in range(len(pl)):
                    rel_rows.append({
                        "model": "gencast", "obs": obs_label,
                        "lead_day": ld, "threshold_mm_day": thr,
                        "forecast_probability": pl[i],
                        "observed_frequency":   of[i],
                        "count":                int(ct[i]),
                        "abs_calibration_gap":  abs_gap[i],
                        "weighted_abs_gap":     weights[i] * abs_gap[i],
                        "ece":                  ece,
                    })
    pd.DataFrame(rel_rows).to_csv(os.path.join(out, "gencast_reliability_tables.csv"), index=False)
    print("  gencast_reliability_tables.csv")

    # 6f. CRPS skill score vs the climatology baseline (needs climatology preds).
    # The climatology baseline is the out-of-sample CHIRPS day-of-year ensemble
    # (see EXPERIMENTAL_SETUP.md), so CRPSS is reported against CHIRPS only to
    # avoid an observational-product mismatch in the denominator.
    if "climatology" in preds:
        crpss_rows = []
        for ld in lead_days_analysis:
            fc_c, obs_c = gather_pairs(preds, "climatology", chirps_2d, init_dates, ld)
            crps_clim = float(np.mean(crps_ensemble(fc_c, obs_c)))
            for m in models:
                fc_m, obs_m = gather_pairs(preds, m, chirps_2d, init_dates, ld)
                crps_m = float(np.mean(crps_ensemble(fc_m, obs_m)))
                crpss = 1.0 - crps_m / crps_clim if crps_clim > 0 else np.nan
                crpss_rows.append({
                    "model": m, "obs": "CHIRPS", "lead_day": ld,
                    "crps_model": round(crps_m, 4),
                    "crps_climatology": round(crps_clim, 4),
                    "crpss": round(crpss, 4),
                })
        pd.DataFrame(crpss_rows).to_csv(
            os.path.join(out, "crpss_vs_climatology_by_model_obs_lead.csv"), index=False)
        print("  crpss_vs_climatology_by_model_obs_lead.csv")
    else:
        print("  [skip CRPSS — no climatology predictions in pred-dir]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config     = BenchmarkConfig()
    INIT_DATES = pd.date_range(args.start, args.end, freq="D")
    LEAD_DAYS  = args.lead_days

    # ── Load data ──
    preds = load_predictions(args.pred_dir, args.models)
    # Climatology baseline (optional) → enables CRPS skill scores vs climatology
    clim_ref = load_climatology_reference(args.pred_dir)
    if clim_ref is not None:
        preds["climatology"] = clim_ref
        print(f"  {'climatology':15s}  {dict(zip(clim_ref.dims, clim_ref.shape))}")
    else:
        print("  climatology      not found — CRPSS vs climatology will be skipped")
    chirps_da, era5_da, tamsat_da = load_observations(config, args.obs_end, args.output_dir)
    (chirps_2d, era5_2d, tamsat_2d,
     chirps_lookup, era5_lookup, tamsat_lookup) = build_lookup_dicts(
        chirps_da, era5_da, tamsat_da)

    LEAD_DAYS_ANALYSIS = [int(x) for x in preds[args.models[0]].lead_day.values]

    # ── Calibration pairs (lead day 1) ──
    print("\nGathering calibration pairs …")
    pairs_chirps = {m: gather_pairs(preds, m, chirps_2d, INIT_DATES, 1) for m in args.models}
    pairs_era5   = {m: gather_pairs(preds, m, era5_2d,   INIT_DATES, 1) for m in args.models}
    pairs_tamsat = {m: gather_pairs(preds, m, tamsat_2d, INIT_DATES, 1) for m in args.models}

    # ── Percentile threshold maps ──
    print("Computing percentile maps …")
    chirps_pctile = compute_pctile_maps(chirps_2d)
    era5_pctile   = compute_pctile_maps(era5_2d)
    tamsat_pctile = compute_pctile_maps(tamsat_2d)

    # ── Figures ──
    plot_timeseries(preds, args.models, INIT_DATES, LEAD_DAYS,
                    chirps_lookup, era5_lookup, tamsat_lookup, args.output_dir)

    plot_temporal_skill(preds, args.models, INIT_DATES, LEAD_DAYS,
                        chirps_2d, args.output_dir)

    plot_rank_histograms(preds, args.models, INIT_DATES, LEAD_DAYS,
                         chirps_2d, era5_2d, tamsat_2d, args.output_dir)

    plot_reliability_local(preds, INIT_DATES,
                           chirps_2d, era5_2d, tamsat_2d,
                           chirps_pctile, era5_pctile, tamsat_pctile,
                           args.output_dir)

    plot_spatial_maps(preds, args.models, INIT_DATES,
                      chirps_2d, era5_2d, LEAD_DAYS, args.output_dir)

    # CRPS skill score vs climatology — maps (only if climatology was loaded)
    if "climatology" in preds:
        print("Plotting CRPSS-vs-climatology maps …")
        plot_crpss_maps(preds, args.models, chirps_2d, INIT_DATES,
                        LEAD_DAYS, args.output_dir, obs_label="chirps")

    # ── CSV tables ──
    compute_and_save_tables(
        preds, args.models, INIT_DATES, LEAD_DAYS_ANALYSIS,
        chirps_2d, era5_2d, tamsat_2d,
        pairs_chirps, pairs_era5, pairs_tamsat,
        args.thresholds, args.output_dir,
    )

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
