// Figure and table catalog, the single source of truth for what the site shows.
// Kept as `window.MANIFEST = <pure JSON>;` so sync_outputs.py can parse and
// validate it against the files actually copied from outputs_2024/.
window.MANIFEST = {
  "figures": [
    {
      "id": "timeseries",
      "tab": "deterministic",
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
      "title": "Spatial bias & RMSE vs CHIRPS, lead day 1",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "1" }
    },
    {
      "id": "spatial_maps_chirps_ld3",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS, lead day 3",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "3" }
    },
    {
      "id": "spatial_maps_chirps_ld5",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS, lead day 5",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "5" }
    },
    {
      "id": "spatial_maps_chirps_ld7",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs CHIRPS, lead day 7",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "CHIRPS", "lead": "7" }
    },
    {
      "id": "spatial_maps_era5_ld1",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5, lead day 1",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "1" }
    },
    {
      "id": "spatial_maps_era5_ld3",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5, lead day 3",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "3" }
    },
    {
      "id": "spatial_maps_era5_ld5",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5, lead day 5",
      "caption": "Per-cell bias (top) and RMSE (bottom) by model; ocean masked (Natural Earth land mask).",
      "tags": { "obs": "ERA5", "lead": "5" }
    },
    {
      "id": "spatial_maps_era5_ld7",
      "tab": "deterministic",
      "title": "Spatial bias & RMSE vs ERA5, lead day 7",
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
      "title": "CRPS skill score vs climatology, scored against CHIRPS",
      "caption": "Per-cell CRPSS relative to the 21-member out-of-sample climatology, model and baseline both scored against CHIRPS; blue beats climatology, black contour = 0, gray = hyper-arid (masked). Rows = models, columns = lead days.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "crpss_maps_tamsat",
      "tab": "climatology",
      "title": "CRPS skill score vs climatology, scored against TAMSAT",
      "caption": "Per-cell CRPSS relative to the 21-member out-of-sample climatology, model and baseline both scored against TAMSAT; blue beats climatology, black contour = 0, gray = hyper-arid (masked). Rows = models, columns = lead days.",
      "tags": { "obs": "TAMSAT" }
    },
    {
      "id": "event_skill_curves_chirps",
      "tab": "events",
      "title": "Event detection skill vs threshold, scored against CHIRPS",
      "caption": "CSI (top), higher is better: of all occasions where rain above the threshold was forecast or observed, the share the model got right (1 = perfect; correct quiet days earn no credit, so rare heavy-rain events score low for every model). Frequency bias (bottom) is alarm-count calibration, not accuracy: events forecast ÷ events observed — the dashed FB = 1 line is ideal, above 1 = over-alerting (crying wolf), below 1 = under-alerting. One column per lead day, from the (ensemble-mean) daily field, scored against CHIRPS; exact numbers in event_scores_by_threshold.csv.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "event_skill_curves_era5",
      "tab": "events",
      "title": "Event detection skill vs threshold, scored against ERA5",
      "caption": "CSI (top), higher is better: of all occasions where rain above the threshold was forecast or observed, the share the model got right (1 = perfect; correct quiet days earn no credit, so rare heavy-rain events score low for every model). Frequency bias (bottom) is alarm-count calibration, not accuracy: events forecast ÷ events observed — the dashed FB = 1 line is ideal, above 1 = over-alerting (crying wolf), below 1 = under-alerting. One column per lead day, from the (ensemble-mean) daily field, scored against ERA5.",
      "tags": { "obs": "ERA5" }
    },
    {
      "id": "event_pod_far_chirps",
      "tab": "events",
      "title": "Event detection vs false alarms (POD / FAR), scored against CHIRPS",
      "caption": "POD (top), higher is better: the share of observed events the model caught (1 = perfect). FAR (bottom), lower is better: the share of the model's alarms that did not verify (0 = perfect). Read the rows together — they are a trade-off: a model can buy a higher POD by alerting more often, at the price of a higher FAR (compare with the frequency-bias row above). One column per lead day, from the (ensemble-mean) daily field, scored against CHIRPS; exact numbers in event_scores_by_threshold.csv.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "event_pod_far_era5",
      "tab": "events",
      "title": "Event detection vs false alarms (POD / FAR), scored against ERA5",
      "caption": "POD (top), higher is better: the share of observed events the model caught (1 = perfect). FAR (bottom), lower is better: the share of the model's alarms that did not verify (0 = perfect). Read the rows together — they are a trade-off: a model can buy a higher POD by alerting more often, at the price of a higher FAR (compare with the frequency-bias row above). One column per lead day, from the (ensemble-mean) daily field, scored against ERA5.",
      "tags": { "obs": "ERA5" }
    },
    {
      "id": "brier_curves_chirps",
      "tab": "events",
      "title": "Brier score by threshold, scored against CHIRPS",
      "caption": "Raw Brier score of the ensemble exceedance probabilities, lower is better; ensemble models only, since deterministic models issue no probabilities. The dashed curve is the climatological base-rate Brier p(1 − p) — the score of always forecasting the observed event frequency — so below it the ensemble adds real information. Brier scores shrink for rarer events by construction: compare models against the dashed reference, not across thresholds. One column per lead day, scored against CHIRPS; exact numbers in brier_scores.csv.",
      "tags": { "obs": "CHIRPS" }
    },
    {
      "id": "brier_curves_era5",
      "tab": "events",
      "title": "Brier score by threshold, scored against ERA5",
      "caption": "Raw Brier score of the ensemble exceedance probabilities, lower is better; ensemble models only, since deterministic models issue no probabilities. The dashed curve is the climatological base-rate Brier p(1 − p) — the score of always forecasting the observed event frequency — so below it the ensemble adds real information. Brier scores shrink for rarer events by construction: compare models against the dashed reference, not across thresholds. One column per lead day, scored against ERA5.",
      "tags": { "obs": "ERA5" }
    }
  ],
  "tables": [
    {
      "id": "event_scores_by_threshold",
      "title": "Event scores (POD / FAR / CSI / frequency bias)",
      "render": false
    },
    {
      "id": "brier_scores",
      "title": "Brier scores by threshold",
      "render": false
    },
    {
      "id": "interval_coverage",
      "title": "Prediction-interval coverage (50 / 80 / 90%)",
      "render": true
    },
    {
      "id": "summary_bias_table",
      "title": "Summary bias / MAE / RMSE (lead day 1)",
      "render": false
    },
    {
      "id": "acc_by_model_truth_lead",
      "title": "Anomaly correlation by model × truth × lead",
      "render": false
    },
    {
      "id": "ssr_by_model_truth_lead",
      "title": "Spread-skill summary (Fortin spread, RMSE, SSR)",
      "render": false
    },
    {
      "id": "crpss_vs_climatology_by_model_obs_lead",
      "title": "CRPS skill score vs climatology",
      "render": false
    },
    {
      "id": "probabilistic_scores",
      "title": "Ensemble CRPS, spread and SSR",
      "render": false
    },
    {
      "id": "deterministic_skill_by_model_obs_lead",
      "title": "Deterministic skill by model × reference × lead",
      "render": false
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
