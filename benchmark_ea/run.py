"""
East Africa benchmark — inference runner (single entry point).

Runs one or more model adapters over a date range and writes one
``pred_YYYY-MM-DD.zarr`` per init date under the output directory. This module
does inference ONLY, verification/scoring lives in ``run_verification.py``.

Quick start
-----------
    # All models, full 2024, lead days 1 3 5 7, precip only (default)
    python -m benchmark_ea.run --start 2024-01-01 --end 2024-12-24

    # Save EVERY variable, ensemble MEAN for non-precip fields (precip kept
    # for all members). Smaller/faster — good default for "all variables".
    python -m benchmark_ea.run --models gencast graphcast fourcastnet \
        --save-variables all --extra-var-members mean

    # Save EVERY variable for EVERY ensemble member (full fidelity, large).
    python -m benchmark_ea.run --models gencast \
        --save-variables all --extra-var-members all

    # Specific window / output location / ensemble size
    python -m benchmark_ea.run --models gencast --start 2024-03-01 --end 2024-05-31 \
        --lead-days 1 3 5 7 --n-members 20 --output-dir data/predictions-allvars

Notes
-----
* All saved variables are regridded/subset to the common East Africa 1° grid.
* Existing complete ``pred_*.zarr`` files are skipped unless --overwrite is set,
  so runs are resumable.
* GPU + the ``aim-graphcast`` conda env are required; launch via
  ``run_inference.sh`` so the environment (LD_LIBRARY_PATH etc.) is set up.
"""

import argparse
import importlib

from benchmark_ea.config import BenchmarkConfig

_ADAPTERS: dict[str, str] = {
    "gencast":     "benchmark_ea.models.gencast:GenCastAdapter",
    "graphcast":   "benchmark_ea.models.graphcast:GraphCastAdapter",
    "fourcastnet": "benchmark_ea.models.fourcastnet:FourCastNetAdapter",
    "climatology": "benchmark_ea.models.climatology:ClimatologyAdapter",
    "neuralgcm":   "benchmark_ea.models.neuralgcm:NeuralGCMAdapter",
}

# Models run by default. NeuralGCM is selectable (--models neuralgcm) but kept
# out of the default set: it needs a dedicated env (separate JAX/dinosaur) and
# isn't installed alongside GraphCast/GenCast.
_DEFAULT_MODELS = ["gencast", "graphcast", "fourcastnet", "climatology"]

# Models that accept a --resolution preset (1.0° small vs 0.25° operational).
# Others (fourcastnet, climatology, neuralgcm) have a single native resolution.
_RES_AWARE = {"graphcast", "gencast"}


def _load_adapter(name: str, resolution: str = "1.0"):
    module_path, cls_name = _ADAPTERS[name].split(":")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    if name in _RES_AWARE:
        return cls(resolution=resolution)
    return cls()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="East Africa AI weather model benchmark — inference runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                   choices=list(_ADAPTERS.keys()),
                   help="Models to run inference for")
    p.add_argument("--resolution", choices=["1.0", "0.25"], default="1.0",
                   help="Checkpoint resolution for resolution-aware models "
                        "(graphcast): '1.0' = small/13-level, '0.25' = "
                        "flagship/37-level. Ignored by other models.")
    p.add_argument("--grid-res", type=float, choices=[1.0, 0.25], default=1.0,
                   help="Output grid (°) the predictions are regridded to before "
                        "saving. Default 1.0. NOTE: --resolution only picks the "
                        "model checkpoint; the SAVED grid is set here, so to keep "
                        "0.25° output you must pass --grid-res 0.25 as well "
                        "(otherwise a 0.25° model is downsampled to 1° on save).")
    p.add_argument("--start", default="2024-01-01", help="First init date (YYYY-MM-DD)")
    p.add_argument("--end",   default="2024-12-24", help="Last init date (YYYY-MM-DD)")
    p.add_argument("--lead-days", nargs="+", type=int, default=[1, 3, 5, 7],
                   metavar="N", help="Forecast lead days to produce")
    p.add_argument("--save-variables", choices=["precip", "all"], default="precip",
                   help="'precip' = daily total_precipitation only; "
                        "'all' = every model variable, regridded to the EA grid")
    p.add_argument("--extra-var-members", choices=["mean", "all"], default="mean",
                   help="For save-variables=all: keep ensemble 'mean' or 'all' "
                        "members for NON-precip vars (precip is always all members)")
    p.add_argument("--n-members", type=int, default=10,
                   help="Ensemble size for stochastic models (GenCast)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write pred_*.zarr (default: <data-dir>/predictions)")
    p.add_argument("--data-dir", default="data",
                   help="Root data dir (for caches and default output location)")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate predictions even if a complete zarr exists")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    config = BenchmarkConfig(
        eval_start        = args.start,
        eval_end          = args.end,
        grid_res          = args.grid_res,
        lead_days         = args.lead_days,
        data_dir          = args.data_dir,
        output_dir        = args.output_dir,
        save_variables    = args.save_variables,
        extra_var_members = args.extra_var_members,
        n_members         = args.n_members,
        overwrite         = args.overwrite,
    )

    sep = "=" * 64
    print(sep)
    print("East Africa benchmark — inference")
    print(f"  dates        : {config.eval_start} … {config.eval_end}")
    print(f"  models       : {', '.join(args.models)}")
    print(f"  resolution   : {args.resolution}° "
          f"(applies to: {', '.join(sorted(_RES_AWARE & set(args.models))) or 'none'})")
    print(f"  lead days    : {config.lead_days}")
    print(f"  save vars    : {config.save_variables}"
          + (f" (extra-var members: {config.extra_var_members})"
             if config.save_variables == "all" else ""))
    print(f"  n_members    : {config.n_members}")
    print(f"  output dir   : {config.predictions_dir}")
    print(sep)

    for model_name in args.models:
        adapter = _load_adapter(model_name, args.resolution)
        head = "=" * 60
        print(f"\n{head}\nModel: {model_name}\n{head}")
        try:
            out_dir = adapter.run_inference(config)
        except (NotImplementedError, ImportError) as exc:
            print(f"  [SKIP — {type(exc).__name__}] {exc}")
            continue
        print(f"  Done → {out_dir}")

    print(f"\nAll inference complete. Predictions in {config.predictions_dir}/")


if __name__ == "__main__":
    main()
