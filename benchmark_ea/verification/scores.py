"""
Verification scoring: forecast/observation pair-gathering and metric aggregation.

Thin layer that pulls matched forecast/obs samples out of the loaded prediction
and truth arrays and applies the pure metric functions from benchmark_ea.metrics
— pooled land-masked pairs for aggregate scores, or per-cell grids for maps.
"""

import numpy as np
import pandas as pd
import xarray as xr

from benchmark_ea import analysis_io
from benchmark_ea.metrics import _crps_per_point

# obs percentiles used for the local-threshold percentile maps
_PCTILE_KEYS = [20, 40, 60, 75, 80]

# The spatial (per-cell, unmasked) pair-gatherer is shared with the analysis
# scripts via benchmark_ea.analysis_io (same signature/behaviour).
gather_pairs_maps = analysis_io.gather_pairs


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


def compute_pctile_maps(obs_2d):
    stack = np.stack(list(obs_2d.values()), axis=0)
    return {q: np.nanpercentile(stack, q, axis=0) for q in _PCTILE_KEYS}


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


def seasonal_mean_field(obs_2d):
    """Per-cell seasonal mean of a truth source — the ACC anomaly reference."""
    return np.nanmean(np.stack(list(obs_2d.values())), axis=0)


def acc_pooled(preds, model, obs_2d, clim_field, init_dates, lead_day):
    """
    Pooled anomaly correlation coefficient over land cells and cases.

        ACC = Σ f'o' / sqrt(Σ f'² · Σ o'²),  f' = fc − clim,  o' = obs − clim

    The forecast is the ensemble mean; anomalies are w.r.t. the per-cell
    seasonal mean of the truth source (``clim_field``).
    """
    fc_all, ob_all = gather_pairs_maps(preds, model, obs_2d, init_dates, lead_day)
    if fc_all is None:
        return np.nan
    fa = fc_all.mean(axis=1) - clim_field[None]
    oa = ob_all - clim_field[None]
    ok = np.isfinite(fa) & np.isfinite(oa)
    fa, oa = fa[ok], oa[ok]
    denom = np.sqrt(np.sum(fa ** 2) * np.sum(oa ** 2))
    return float(np.sum(fa * oa) / denom) if denom > 0 else np.nan


def spread_skill_pooled(preds, model, obs_2d, init_dates, lead_day):
    """
    Pooled (cells × cases) ensemble spread, RMSE and their ratio at one lead.

        spread = sqrt( (M+1)/M · <ensemble variance> )     Fortin et al. (2014)
        SSR    = spread / RMSE(ensemble mean)              1 = well calibrated

    Returns (spread, rmse, ssr).
    """
    fc_all, ob_all = gather_pairs_maps(preds, model, obs_2d, init_dates, lead_day)
    if fc_all is None:
        return np.nan, np.nan, np.nan
    m = fc_all.shape[1]
    corr = (m + 1) / m
    err = fc_all.mean(axis=1) - ob_all
    var = fc_all.var(axis=1, ddof=1)
    land = np.isfinite(ob_all)
    rmse = float(np.sqrt(np.mean(err[land] ** 2)))
    spread = float(np.sqrt(corr * np.mean(var[land])))
    return spread, rmse, spread / rmse if rmse > 0 else np.nan


def ssr_by_lat(preds, model, obs_2d, init_dates, lead_day):
    """Per-latitude SSR (variance and squared error pooled over lon and cases)."""
    fc_all, ob_all = gather_pairs_maps(preds, model, obs_2d, init_dates, lead_day)
    if fc_all is None:
        return None
    m = fc_all.shape[1]
    corr = (m + 1) / m
    err = fc_all.mean(axis=1) - ob_all
    var = fc_all.var(axis=1, ddof=1)
    land = np.isfinite(ob_all)
    err2 = np.where(land, err ** 2, np.nan)
    var = np.where(land, var, np.nan)
    mean_err2 = np.nanmean(np.nanmean(err2, axis=0), axis=1)   # cases → lon → (lat,)
    mean_var = np.nanmean(np.nanmean(var, axis=0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(corr * mean_var) / np.sqrt(mean_err2)


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
