# East Africa AI Weather-Model Benchmark

Benchmarks machine-learning weather forecast systems — **GenCast** (diffusion
ensemble), **NeuralGCM** (hybrid dynamical-ML ensemble), **GraphCast**
(deterministic GNN), **FourCastNet v2 + PrecipitationAFNO** (deterministic
SFNO) — against a **climatology baseline** over East Africa, verified against
**CHIRPS**, **ERA5**, and **TAMSAT**.

See [EXPERIMENTAL_SETUP.md](EXPERIMENTAL_SETUP.md) for domain, models, references,
and metrics.

## Layout

```
benchmark_ea/            inference package
  run.py                 single inference entry point (CLI)
  config.py              BenchmarkConfig (grid, dates, saving options)
  regrid.py              shared regrid/subset to the EA 1° grid
  models/                one adapter per model (gencast, graphcast, fourcastnet, neuralgcm, climatology)
  truth/                 observation loaders (chirps, era5, tamsat)
  verification/          scoring, publication figures (style.py) and CSV tables
run_verification.py      verification entry point → outputs_2024/
run_inference.sh         env wrapper (single GPU); forwards all args to benchmark_ea.run
run_inference_parallel.sh  multi-GPU launcher (one model per GPU)
run_neuralgcm.sh         NeuralGCM launcher (own conda env)
```

## Environment

Two conda environments are used, pinned as lockfiles under `envs/`:

```bash
# GenCast / GraphCast / FourCastNet v2 + verification
conda env create -f envs/aim-graphcast.yml
conda activate aim-graphcast
pip install -e .            # install this repo (benchmark-ea)

# NeuralGCM runs in its own env (incompatible JAX/dinosaur stack)
conda env create -f envs/neuralgcm.yml
```

Both `.yml` files are exact-version snapshots of the working environments (no
build strings). The core Python deps are also lower-bound pinned in
`pyproject.toml`; the `envs/` lockfiles are the authoritative reproducible spec.
A few ML packages (graphcast, earth2mip, modulus, jraph, dinosaur) are installed
from Git/source — see the header notes in each `envs/*.yml`.

## Running inference

Inference is driven by the single entry point `benchmark_ea.run`. Launch it
through `run_inference.sh`, which activates the `aim-graphcast` conda env and
sets up `LD_LIBRARY_PATH`:

```bash
# Precipitation only (default) — smallest output
./run_inference.sh --models gencast graphcast fourcastnet climatology \
    --start 2024-01-01 --end 2024-12-24 --lead-days 1 3 5 7
```

### Saving all variables

The same script saves every model variable, regridded/subset to the EA 1° grid,
via `--save-variables all`. For ensemble models, `--extra-var-members` controls
how many members are kept for the **non-precip** fields (precipitation is always
kept for every member):

```bash
# All variables, ensemble MEAN for non-precip fields (precip = all members)
./run_inference.sh --models gencast graphcast fourcastnet \
    --save-variables all --extra-var-members mean

# All variables, EVERY ensemble member (full fidelity, large)
./run_inference.sh --models gencast \
    --save-variables all --extra-var-members all
```

Key flags: `--models --start --end --lead-days --save-variables
--extra-var-members --n-members --output-dir --overwrite`. Run
`./run_inference.sh --help` for the full list.

Multi-GPU (one model per GPU) — same options via environment variables:

```bash
SAVE_VARIABLES=all EXTRA_VAR_MEMBERS=mean ./run_inference_parallel.sh
```

Each init date is written to `<output-dir>/<model>/pred_YYYY-MM-DD.zarr`
(default `data/predictions`). Complete files are skipped unless `--overwrite`.

## Verification

```bash
python run_verification.py --pred-dir data/predictions \
    --start 2024-03-01 --end 2024-05-31 \
    --models fourcastnet gencast graphcast neuralgcm
```

One command produces every publication figure (vector PDF + 300-dpi PNG; a
colorblind-validated palette, defined once in
`benchmark_ea/verification/style.py`) and every CSV table. Ensemble-only
diagnostics (CRPS/spread, rank histograms, reliability) automatically cover
every model with more than one member — currently GenCast and NeuralGCM.
Verification reads only `total_precipitation`, so it works on both precip-only
and all-variable prediction stores.

**Climatology baseline / CRPSS.** If `<pred-dir>/climatology/` exists, the
climatology 21-member baseline is loaded automatically and used as the reference
for the CRPS skill score:

* `crpss_vs_climatology_by_model_obs_lead.csv` — CRPSS per model × reference × lead
* `crpss_maps_chirps.pdf/.png` — per-cell CRPSS maps, models × lead days

`CRPSS = 1 - CRPS_model / CRPS_climatology` (> 0 = beats climatology). Generate
the baseline first:

```bash
./run_inference.sh --models climatology --start 2024-01-01 --end 2024-12-24 \
    --output-dir data/predictions
```

## Tests

```bash
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/aim-graphcast/bin/python \
    -m pytest tests/ -q --assert=plain -p no:cacheprovider
```

## Notes

* `data/predictions/` holds the current full-year 2024 run (359 inits per
  model, written 2026-07-11) on the common 1° East Africa grid. If you ever
  regenerate with changed grid settings, clear or archive the old files first —
  `run_verification.py` will refuse to mix grids.
