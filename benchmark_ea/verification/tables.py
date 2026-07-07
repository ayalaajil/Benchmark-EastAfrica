"""
Verification CSV tables: deterministic skill, event scores, probabilistic
scores for every ensemble model, reliability, ACC / spread-skill summaries,
and CRPSS vs the climatology baseline.
"""

import os

import numpy as np
import pandas as pd

from benchmark_ea.metrics import (
    deterministic_metrics,
    contingency_scores,
    crps_ensemble,
    brier_score_ensemble,
    interval_coverage,
    reliability_diagram,
)
from benchmark_ea.verification.scores import (
    acc_pooled,
    gather_pairs,
    seasonal_mean_field,
    spread_skill_pooled,
)


def compute_and_save_tables(preds, models, ens_models, init_dates,
                            lead_days_analysis,
                            chirps_2d, era5_2d, tamsat_2d,
                            pairs_chirps, pairs_era5, pairs_tamsat,
                            thresholds, out):
    print("\n[11] Computing and saving CSV tables …")

    # 11a. Summary bias table (lead day 1)
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

    # 11b. Deterministic skill across all lead days
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

    # 11c. Event-based scores
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

    # 11d. Probabilistic scores for every ensemble model
    prob_rows, cov_rows, brier_rows = [], [], []
    for m in ens_models:
        for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
            for ld in lead_days_analysis:
                fc_ens, obs = gather_pairs(preds, m, obs_2d, init_dates, ld)
                crps    = crps_ensemble(fc_ens, obs)
                fc_mean = fc_ens.mean(axis=1)
                spread  = fc_ens.std(axis=1, ddof=1)
                rmse    = float(np.sqrt(np.mean((fc_mean - obs) ** 2)))
                prob_rows.append({
                    "model": m, "obs": obs_label, "lead_day": ld,
                    "n":                    len(obs),
                    "mean_crps":            float(np.mean(crps)),
                    "rmse_ensemble_mean":   rmse,
                    "mean_ensemble_spread": float(np.mean(spread)),
                    "spread_skill_ratio":   float(np.mean(spread)) / rmse if rmse > 0 else np.nan,
                })
                for nominal in [0.50, 0.80, 0.90]:
                    row = interval_coverage(fc_ens, obs, nominal)
                    row.update({"model": m, "obs": obs_label, "lead_day": ld})
                    cov_rows.append(row)
                for thr in thresholds:
                    brier_rows.append({
                        "model": m, "obs": obs_label,
                        "lead_day": ld, "threshold_mm_day": thr,
                        "brier_score": brier_score_ensemble(fc_ens, obs, thr),
                        "event_rate":  float(np.mean(obs > thr)),
                    })
    pd.DataFrame(prob_rows).to_csv(os.path.join(out, "probabilistic_scores.csv"),  index=False)
    pd.DataFrame(cov_rows).to_csv(os.path.join(out, "interval_coverage.csv"),      index=False)
    pd.DataFrame(brier_rows).to_csv(os.path.join(out, "brier_scores.csv"),         index=False)
    print("  probabilistic_scores.csv / interval_coverage.csv / brier_scores.csv")

    # 11e. Reliability tables + ECE for every ensemble model
    rel_rows = []
    for m in ens_models:
        for obs_label, obs_2d in [("CHIRPS", chirps_2d), ("ERA5", era5_2d)]:
            for ld in lead_days_analysis:
                fc_ens, obs = gather_pairs(preds, m, obs_2d, init_dates, ld)
                for thr in thresholds:
                    pl, of, ct = reliability_diagram(fc_ens, obs, thr)
                    weights    = ct / ct.sum() if ct.sum() > 0 else np.zeros_like(ct, float)
                    abs_gap    = np.abs(of - pl)
                    ece        = float(np.nansum(weights * abs_gap))
                    for i in range(len(pl)):
                        rel_rows.append({
                            "model": m, "obs": obs_label,
                            "lead_day": ld, "threshold_mm_day": thr,
                            "forecast_probability": pl[i],
                            "observed_frequency":   of[i],
                            "count":                int(ct[i]),
                            "abs_calibration_gap":  abs_gap[i],
                            "weighted_abs_gap":     weights[i] * abs_gap[i],
                            "ece":                  ece,
                        })
    pd.DataFrame(rel_rows).to_csv(os.path.join(out, "reliability_tables.csv"), index=False)
    print("  reliability_tables.csv")

    # 11f. ACC and spread-skill summaries (match the lead-curve figures)
    acc_rows = []
    for t_label, obs_2d in [("CHIRPS", chirps_2d), ("TAMSAT", tamsat_2d)]:
        clim_field = seasonal_mean_field(obs_2d)
        for m in models:
            for ld in lead_days_analysis:
                acc_rows.append({
                    "model": m, "truth": t_label, "lead_day": ld,
                    "acc": round(acc_pooled(preds, m, obs_2d, clim_field,
                                            init_dates, ld), 4),
                })
    pd.DataFrame(acc_rows).to_csv(os.path.join(out, "acc_by_model_truth_lead.csv"), index=False)
    print("  acc_by_model_truth_lead.csv")

    ssr_rows = []
    for t_label, obs_2d in [("CHIRPS", chirps_2d), ("TAMSAT", tamsat_2d)]:
        for m in ens_models:
            for ld in lead_days_analysis:
                spread, rmse, ssr = spread_skill_pooled(preds, m, obs_2d,
                                                        init_dates, ld)
                ssr_rows.append({
                    "model": m, "truth": t_label, "lead_day": ld,
                    "spread_fortin": round(spread, 4),
                    "rmse_ens_mean": round(rmse, 4),
                    "ssr":           round(ssr, 4),
                })
    pd.DataFrame(ssr_rows).to_csv(os.path.join(out, "ssr_by_model_truth_lead.csv"), index=False)
    print("  ssr_by_model_truth_lead.csv")

    # 11g. CRPS skill score vs the climatology baseline (needs climatology preds).
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
