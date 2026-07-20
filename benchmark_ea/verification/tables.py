"""
Verification CSV tables: deterministic skill, event scores, probabilistic
scores for every ensemble model, reliability, ACC / spread-skill summaries,
and CRPSS vs the climatology baseline.

Every table is stratified by season (``benchmark_ea.verification.seasons``)
in addition to the annual aggregate — East Africa's rainfall is strongly
bimodal (the MAM "long rains" and OND "short rains"), and annual-mean skill
can hide season-dependent behaviour. ``season == "annual"`` is the full-period
aggregate (the only row previously computed); MAM/JJAS/OND/JF restrict each
row to cases whose *valid* date falls in that season.

Rather than one CSV per table type with a mixed ``season`` column, each
table is written once per season into its own ``<out>/<season>/`` subfolder
(``<out>/annual/``, ``<out>/MAM/``, …) — a self-contained report per period,
so browsing one season's results doesn't mean filtering a combined file. The
``season`` column is kept in each file too (redundant within one file, but
self-descriptive if it's copied out of its folder).

All spatially pooled aggregates (bias/MAE/RMSE, CRPS, Brier, interval
coverage, spread-skill, ACC) use cos(latitude) area weights, so regional means
are not biased toward the higher-latitude cells of the domain, per
EXPERIMENTAL_SETUP.md. Contingency-table counts (hits/misses/false alarms)
are intentionally left unweighted — an area-weighted count is not a
meaningful integer quantity; only ratios of counts are reported.
"""

import os

import numpy as np
import pandas as pd

from benchmark_ea.metrics import (
    deterministic_metrics,
    contingency_scores,
    crps_ensemble,
    brier_score_ensemble,
    brier_decomposition,
    interval_coverage,
    rank_histogram,
    rank_histogram_flatness,
    reliability_diagram,
    spread_skill,
)
from benchmark_ea.verification.scores import (
    acc_pooled,
    gather_pairs,
    lat_weight_grid,
    seasonal_mean_field,
    spread_skill_pooled,
)
from benchmark_ea.verification.seasons import SEASONS, filter_by_season


def _save_by_season(df, out, filename):
    """Write ``df`` (which has a ``season`` column spanning all of SEASONS)
    as one CSV per season, into ``<out>/<season>/<filename>``. If ``df`` has
    no rows at all (no "season" column to filter on), an empty CSV is
    written to every season folder rather than raising."""
    for season in SEASONS:
        season_dir = os.path.join(out, season)
        os.makedirs(season_dir, exist_ok=True)
        sub = df[df["season"] == season] if "season" in df.columns else df
        sub.to_csv(os.path.join(season_dir, filename), index=False)


def compute_and_save_tables(preds, models, ens_models, init_dates,
                            lead_days_analysis,
                            chirps_2d, era5_2d, tamsat_2d,
                            thresholds, out):
    print("\n[11] Computing and saving CSV tables …")

    obs_sources = [("CHIRPS", chirps_2d), ("ERA5", era5_2d), ("TAMSAT", tamsat_2d)]
    # The area-weight grid depends only on the (shared) target grid, not on
    # model or lead day, so one grid serves every call below.
    w_grid = lat_weight_grid(preds, models[0], lead_days_analysis[0])

    def _gather(m, obs_2d, lead_day, season):
        """gather_pairs (fc_ens, obs, cos-lat weights) for one
        (model, obs, lead, season); season="annual" gathers the full period."""
        season_arg = None if season == "annual" else season
        return gather_pairs(preds, m, obs_2d, init_dates, lead_day,
                            return_weights=True, season=season_arg)

    # 11a. Summary bias table (lead day 1)
    rows = []
    for season in SEASONS:
        for m in models:
            for obs_label, obs_2d in obs_sources:
                fc_ens, obs, w = _gather(m, obs_2d, 1, season)
                if fc_ens is None:
                    continue
                d = deterministic_metrics(fc_ens.mean(axis=1), obs, weights=w)
                rows.append({
                    "model": m, "vs": obs_label, "season": season,
                    "bias (mm/d)": round(d["bias"], 3),
                    "MAE (mm/d)":  round(d["mae"], 3),
                    "RMSE (mm/d)": round(d["rmse"], 3),
                })
    _save_by_season(pd.DataFrame(rows), out, "summary_bias_table.csv")
    print("  summary_bias_table.csv")

    # 11b. Deterministic skill across all lead days
    skill_rows = []
    for season in SEASONS:
        for obs_label, obs_2d in obs_sources:
            for ld in lead_days_analysis:
                for m in models:
                    fc_ens, obs, w = _gather(m, obs_2d, ld, season)
                    if fc_ens is None:
                        continue
                    row = deterministic_metrics(fc_ens.mean(axis=1), obs, weights=w)
                    row.update({"model": m, "obs": obs_label, "lead_day": ld,
                               "season": season})
                    skill_rows.append(row)
    _save_by_season(pd.DataFrame(skill_rows), out, "deterministic_skill_by_model_obs_lead.csv")
    print("  deterministic_skill_by_model_obs_lead.csv")

    # 11c. Event-based scores (contingency counts are unweighted, see module docstring)
    event_rows = []
    for season in SEASONS:
        for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
            for ld in lead_days_analysis:
                for m in models:
                    fc_ens, obs, _w = _gather(m, obs_2d, ld, season)
                    if fc_ens is None:
                        continue
                    fc_mean = fc_ens.mean(axis=1)
                    for thr in thresholds:
                        row = contingency_scores(fc_mean, obs, thr)
                        row.update({"model": m, "obs": obs_label,
                                    "lead_day": ld, "threshold_mm_day": thr,
                                    "season": season})
                        event_rows.append(row)
    _save_by_season(pd.DataFrame(event_rows), out, "event_scores_by_threshold.csv")
    print("  event_scores_by_threshold.csv")

    # 11d. Probabilistic scores for every ensemble model
    prob_rows, cov_rows, brier_rows = [], [], []
    for season in SEASONS:
        for m in ens_models:
            for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
                for ld in lead_days_analysis:
                    fc_ens, obs, w = _gather(m, obs_2d, ld, season)
                    if fc_ens is None:
                        continue
                    crps = crps_ensemble(fc_ens, obs)          # fair (Ferro 2014)
                    spread, rmse, ssr = spread_skill(fc_ens, obs, weights=w)
                    prob_rows.append({
                        "model": m, "obs": obs_label, "lead_day": ld, "season": season,
                        "n":                    len(obs),
                        "mean_crps":            float(np.average(crps, weights=w)),
                        "rmse_ensemble_mean":   rmse,
                        "mean_ensemble_spread": spread,
                        "spread_skill_ratio":   ssr,
                    })
                    for nominal in [0.50, 0.80, 0.90]:
                        row = interval_coverage(fc_ens, obs, nominal, weights=w)
                        row.update({"model": m, "obs": obs_label, "lead_day": ld,
                                   "season": season})
                        cov_rows.append(row)
                    for thr in thresholds:
                        dec = brier_decomposition(fc_ens, obs, thr)
                        brier_rows.append({
                            "model": m, "obs": obs_label,
                            "lead_day": ld, "threshold_mm_day": thr, "season": season,
                            "brier_score": brier_score_ensemble(fc_ens, obs, thr, weights=w),
                            "event_rate":  float(np.average((obs > thr).astype(float), weights=w)),
                            "bss":         dec["bss"],
                            "reliability": dec["reliability"],
                            "resolution":  dec["resolution"],
                            "uncertainty": dec["uncertainty"],
                        })
    _save_by_season(pd.DataFrame(prob_rows), out, "probabilistic_scores.csv")
    _save_by_season(pd.DataFrame(cov_rows), out, "interval_coverage.csv")
    _save_by_season(pd.DataFrame(brier_rows), out, "brier_scores.csv")
    print("  probabilistic_scores.csv / interval_coverage.csv / brier_scores.csv")

    # 11e. Reliability tables + ECE for every ensemble model
    rel_rows = []
    for season in SEASONS:
        for m in ens_models:
            for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
                for ld in lead_days_analysis:
                    fc_ens, obs, _w = _gather(m, obs_2d, ld, season)
                    if fc_ens is None:
                        continue
                    for thr in thresholds:
                        pl, of, ct = reliability_diagram(fc_ens, obs, thr)
                        weights    = ct / ct.sum() if ct.sum() > 0 else np.zeros_like(ct, float)
                        abs_gap    = np.abs(of - pl)
                        ece        = float(np.nansum(weights * abs_gap))
                        for i in range(len(pl)):
                            rel_rows.append({
                                "model": m, "obs": obs_label,
                                "lead_day": ld, "threshold_mm_day": thr, "season": season,
                                "forecast_probability": pl[i],
                                "observed_frequency":   of[i],
                                "count":                int(ct[i]),
                                "abs_calibration_gap":  abs_gap[i],
                                "weighted_abs_gap":     weights[i] * abs_gap[i],
                                "ece":                  ece,
                            })
    _save_by_season(pd.DataFrame(rel_rows), out, "reliability_tables.csv")
    print("  reliability_tables.csv")

    # 11f. ACC and spread-skill summaries (match the lead-curve figures)
    acc_rows = []
    for season in SEASONS:
        for t_label, obs_2d in [("CHIRPS", chirps_2d), ("TAMSAT", tamsat_2d)]:
            clim_field = seasonal_mean_field(obs_2d)
            for m in models:
                for ld in lead_days_analysis:
                    season_dates = filter_by_season(init_dates, ld, season)
                    acc = acc_pooled(preds, m, obs_2d, clim_field, season_dates, ld,
                                     weights=w_grid)
                    acc_rows.append({
                        "model": m, "truth": t_label, "lead_day": ld, "season": season,
                        "acc": round(acc, 4) if np.isfinite(acc) else acc,
                    })
    _save_by_season(pd.DataFrame(acc_rows), out, "acc_by_model_truth_lead.csv")
    print("  acc_by_model_truth_lead.csv")

    ssr_rows = []
    for season in SEASONS:
        for t_label, obs_2d in [("CHIRPS", chirps_2d), ("TAMSAT", tamsat_2d)]:
            for m in ens_models:
                for ld in lead_days_analysis:
                    season_dates = filter_by_season(init_dates, ld, season)
                    spread, rmse, ssr = spread_skill_pooled(preds, m, obs_2d,
                                                            season_dates, ld,
                                                            weights=w_grid)
                    ssr_rows.append({
                        "model": m, "truth": t_label, "lead_day": ld, "season": season,
                        "spread_fortin": round(spread, 4) if np.isfinite(spread) else spread,
                        "rmse_ens_mean": round(rmse, 4) if np.isfinite(rmse) else rmse,
                        "ssr":           round(ssr, 4) if np.isfinite(ssr) else ssr,
                    })
    _save_by_season(pd.DataFrame(ssr_rows), out, "ssr_by_model_truth_lead.csv")
    print("  ssr_by_model_truth_lead.csv")

    # 11g. CRPS skill score vs the climatology baseline (needs climatology preds).
    # The climatology baseline is the out-of-sample CHIRPS day-of-year ensemble
    # (see EXPERIMENTAL_SETUP.md), so CRPSS is reported against CHIRPS only to
    # avoid an observational-product mismatch in the denominator.
    if "climatology" in preds:
        crpss_rows = []
        for season in SEASONS:
            for ld in lead_days_analysis:
                season_arg = None if season == "annual" else season
                fc_c, obs_c, w_c = gather_pairs(preds, "climatology", chirps_2d,
                                                init_dates, ld, return_weights=True,
                                                season=season_arg)
                if fc_c is None:
                    continue
                crps_clim = float(np.average(crps_ensemble(fc_c, obs_c), weights=w_c))
                for m in models:
                    fc_m, obs_m, w_m = gather_pairs(preds, m, chirps_2d, init_dates, ld,
                                                    return_weights=True, season=season_arg)
                    if fc_m is None:
                        continue
                    crps_m = float(np.average(crps_ensemble(fc_m, obs_m), weights=w_m))
                    crpss = 1.0 - crps_m / crps_clim if crps_clim > 0 else np.nan
                    crpss_rows.append({
                        "model": m, "obs": "CHIRPS", "lead_day": ld, "season": season,
                        "crps_model": round(crps_m, 4),
                        "crps_climatology": round(crps_clim, 4),
                        "crpss": round(crpss, 4) if np.isfinite(crpss) else crpss,
                    })
        _save_by_season(pd.DataFrame(crpss_rows), out, "crpss_vs_climatology_by_model_obs_lead.csv")
        print("  crpss_vs_climatology_by_model_obs_lead.csv")
    else:
        print("  [skip CRPSS — no climatology predictions in pred-dir]")

    # 11h. Rank-histogram flatness (Talagrand calibration summary) — delta = 0
    # is a perfectly flat (calibrated) histogram; see rank_histogram_flatness.
    rank_rows = []
    for season in SEASONS:
        for m in ens_models:
            for obs_label, obs_2d in obs_sources:
                for ld in lead_days_analysis:
                    fc_ens, obs, _w = _gather(m, obs_2d, ld, season)
                    if fc_ens is None:
                        continue
                    freq = rank_histogram(fc_ens, obs)
                    rank_rows.append({
                        "model": m, "obs": obs_label, "lead_day": ld, "season": season,
                        "n": len(obs),
                        "delta_flatness": rank_histogram_flatness(freq),
                    })
    _save_by_season(pd.DataFrame(rank_rows), out, "rank_histogram_stats.csv")
    print("  rank_histogram_stats.csv")
