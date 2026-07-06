// Figure and table catalog — the single source of truth for what the site shows.
// Kept as `window.MANIFEST = <pure JSON>;` so sync_outputs.py can parse and
// validate it against the files actually copied from mam2024_analysis_outputs/.
window.MANIFEST = {
  "figures": [
    {
      "id": "timeseries",
      "tab": "overview",
      "title": "Area-mean daily precipitation",
      "caption": "Land-only domain mean, one panel per lead day; four AI models against CHIRPS, ERA5 and TAMSAT.",
      "tags": {}
    },
    {
      "id": "temporal_bias_mae",
      "tab": "deterministic",
      "title": "Temporal bias and MAE",
      "caption": "7-day rolling bias (top) and MAE (bottom) vs CHIRPS, one column per lead day.",
      "tags": {}
    },
    {
      "id": "acc_lead_curves",
      "tab": "deterministic",
      "title": "Anomaly correlation vs lead day",
      "caption": "Pooled ACC against CHIRPS (a) and TAMSAT (b); anomalies w.r.t. each truth's seasonal mean.",
      "tags": {}
    },
    {
      "id": "spatial_maps_chirps_ld1",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS — lead day 1",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "1" }
    },
    {
      "id": "spatial_maps_chirps_ld3",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS — lead day 3",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "3" }
    },
    {
      "id": "spatial_maps_chirps_ld5",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS — lead day 5",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "5" }
    },
    {
      "id": "spatial_maps_chirps_ld7",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS — lead day 7",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "7" }
    },
    {
      "id": "spatial_maps_era5_ld1",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5 — lead day 1",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "1" }
    },
    {
      "id": "spatial_maps_era5_ld3",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5 — lead day 3",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "3" }
    },
    {
      "id": "spatial_maps_era5_ld5",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5 — lead day 5",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "5" }
    },
    {
      "id": "spatial_maps_era5_ld7",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5 — lead day 7",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "7" }
    },
    {
      "id": "zonal_skill_profiles",
      "tab": "deterministic",
      "title": "Zonal skill profiles (lead day 1)",
      "caption": "CRPS, bias, MAE and RMSE by latitude. Legacy figure from skill_analysis.ipynb (PNG only).",
      "tags": {},
      "no_pdf": true
    },
    {
      "id": "zonal_skill_profiles_all_leads",
      "tab": "deterministic",
      "title": "Zonal skill profiles (all lead days)",
      "caption": "CRPS, bias, MAE and RMSE by latitude for every lead day. Legacy figure from skill_analysis.ipynb (PNG only).",
      "tags": {},
      "no_pdf": true
    },
    {
      "id": "ensemble_crps_spread_ssr",
      "tab": "probabilistic",
      "title": "Ensemble CRPS, spread and spread/skill",
      "caption": "Season-long CRPS, ensemble spread and spread/RMSE (7-day rolling); rows = GenCast, NeuralGCM; one curve per lead day.",
      "tags": {}
    },
    {
      "id": "rank_histograms_gencast",
      "tab": "probabilistic",
      "title": "GenCast rank histograms",
      "caption": "Tie-aware Talagrand ranks (Hamill 2001) vs CHIRPS, ERA5 and TAMSAT across lead days.",
      "tags": { "model": "GenCast" }
    },
    {
      "id": "rank_histograms_neuralgcm",
      "tab": "probabilistic",
      "title": "NeuralGCM rank histograms",
      "caption": "Tie-aware Talagrand ranks (Hamill 2001) vs CHIRPS, ERA5 and TAMSAT across lead days.",
      "tags": { "model": "NeuralGCM" }
    },
    {
      "id": "reliability_gencast",
      "tab": "probabilistic",
      "title": "GenCast reliability diagrams",
      "caption": "Local percentile thresholds (P20–P80), lead day 1, with 95% Wilson intervals and per-panel ECE.",
      "tags": { "model": "GenCast" }
    },
    {
      "id": "reliability_neuralgcm",
      "tab": "probabilistic",
      "title": "NeuralGCM reliability diagrams",
      "caption": "Local percentile thresholds (P20–P80), lead day 1, with 95% Wilson intervals and per-panel ECE.",
      "tags": { "model": "NeuralGCM" }
    },
    {
      "id": "ssr_lead_curves_chirps",
      "tab": "probabilistic",
      "title": "Spread-skill vs lead day (truth CHIRPS)",
      "caption": "Spread/RMSE ratio (a) and its ingredients (b) for both ensembles; Fortin-corrected spread.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "ssr_lead_curves_tamsat",
      "tab": "probabilistic",
      "title": "Spread-skill vs lead day (truth TAMSAT)",
      "caption": "Spread/RMSE ratio (a) and its ingredients (b) for both ensembles; Fortin-corrected spread.",
      "tags": { "obs": "TAMSAT" }
    },
    {
      "id": "ssr_zonal_chirps",
      "tab": "probabilistic",
      "title": "Zonal spread-skill profile (truth CHIRPS)",
      "caption": "Spread/RMSE by latitude for GenCast (a) and NeuralGCM (b), one curve per lead day.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "ssr_zonal_tamsat",
      "tab": "probabilistic",
      "title": "Zonal spread-skill profile (truth TAMSAT)",
      "caption": "Spread/RMSE by latitude for GenCast (a) and NeuralGCM (b), one curve per lead day.",
      "tags": { "obs": "TAMSAT" }
    },
    {
      "id": "crpss_maps_chirps",
      "tab": "climatology",
      "title": "CRPS skill score vs climatology",
      "caption": "Per-cell CRPSS against the 21-member out-of-sample CHIRPS climatology; blue beats climatology, black contour = 0, gray = hyper-arid (masked). Rows = models, columns = lead days.",
      "tags": {}
    }
  ],
  "tables": [
    {
      "id": "summary_bias_table",
      "title": "Summary bias / MAE / RMSE (lead day 1)",
      "render": true
    },
    {
      "id": "acc_by_model_truth_lead",
      "title": "Anomaly correlation by model × truth × lead",
      "render": true
    },
    {
      "id": "ssr_by_model_truth_lead",
      "title": "Spread-skill summary (Fortin spread, RMSE, SSR)",
      "render": true
    },
    {
      "id": "crpss_vs_climatology_by_model_obs_lead",
      "title": "CRPS skill score vs climatology",
      "render": true
    },
    {
      "id": "probabilistic_scores",
      "title": "Ensemble CRPS, spread and SSR",
      "render": true
    },
    {
      "id": "interval_coverage",
      "title": "Prediction-interval coverage (50 / 80 / 90%)",
      "render": true
    },
    {
      "id": "brier_scores",
      "title": "Brier scores by threshold",
      "render": true
    },
    {
      "id": "deterministic_skill_by_model_obs_lead",
      "title": "Deterministic skill by model × reference × lead",
      "render": true
    },
    {
      "id": "event_scores_by_threshold",
      "title": "Event scores (POD / FAR / CSI / frequency bias)",
      "render": true
    },
    {
      "id": "reliability_tables",
      "title": "Full reliability-diagram bins (large)",
      "render": false
    },
    {
      "id": "gencast_reliability_ece",
      "title": "GenCast reliability ECE (legacy notebook output)",
      "render": false
    },
    {
      "id": "gencast_spread_skill_by_bin",
      "title": "GenCast spread-skill by bin (legacy notebook output)",
      "render": false
    },
    {
      "id": "regional_skill_by_lat_band",
      "title": "Regional skill by latitude band (legacy notebook output)",
      "render": false
    },
    {
      "id": "data_coverage_and_range_checks",
      "title": "Data coverage and range checks",
      "render": false
    },
    {
      "id": "valid_date_coverage_by_lead",
      "title": "Valid-date coverage by lead",
      "render": false
    }
  ]
};
