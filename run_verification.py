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
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_ea.analysis_io import load_predictions
from benchmark_ea.config import BenchmarkConfig
from benchmark_ea.verification.data import (
    build_lookup_dicts,
    load_climatology_reference,
    load_observations,
)
from benchmark_ea.verification.scores import (
    compute_pctile_maps,
    crpss_maps_vs_climatology,   # noqa: F401 — re-exported for the standalone CRPSS scripts
    gather_pairs,
)
from benchmark_ea.verification.plots import (
    plot_crpss_maps,
    plot_rank_histograms,
    plot_reliability_local,
    plot_spatial_maps,
    plot_temporal_skill,
    plot_timeseries,
)
from benchmark_ea.verification.tables import compute_and_save_tables


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
