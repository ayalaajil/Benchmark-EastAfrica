# Experimental Setup

## Goal

We benchmark the leading families of machine-learning weather models on a single,
operationally relevant target: **daily total rainfall over East Africa**. The
benchmark is deliberately self-contained and reproducible — every input (model
checkpoints, ERA5 initial conditions, and observational references) is public,
and inference for all models is driven by one pipeline that writes a common,
model-agnostic output format. The scientific question is whether — and at which
lead times, seasons, and rainfall intensities — modern AI forecast systems add
skill over a strong climatological baseline for precipitation in a data-sparse,
convectively-dominated tropical region.

## Study domain and evaluation grid

We evaluate on a common regular grid spanning **12°S–15°N and 28°E–52°E at 1°
resolution** (28 × 25 = 700 grid cells), which matches the native output
resolution of the graph-based models and serves as the lowest common denominator
across all systems. Verification is restricted to land using a mask derived from
the Natural Earth 50 m land polygons (~512 land cells); ocean and large
inland-lake cells are excluded throughout. For regional analysis we further
subset the domain into the seven flood-season countries of the Greater Horn of
Africa — Kenya, Ethiopia, Tanzania, Somalia, Uganda, Rwanda, and Burundi — and
into zonal latitude bands. All spatially aggregated statistics use cos(latitude)
area weights, so regional means are not biased toward the higher latitudes of the
domain.

## Evaluation period and forecast protocol

We initialize each model **daily over the whole of 2024**, from 1 January to 24
December — 359 initialization dates (the last init is 24 December so that a
7-day forecast still verifies within the year). Covering the full year, rather
than a single season, lets us characterize skill across East Africa's bimodal
rainfall regime — the March–May "long rains" and the October–December "short
rains" — as well as the intervening drier months, and to test whether AI skill
is season-dependent.

Each model is initialized at **00 UTC** from ERA5 analyses (drawn from the public
ARCO-ERA5 archive) and integrated forward to the maximum required lead time. We
verify at lead times of **1, 3, 5, and 7 days**: a forecast initialized on date
*t* is scored against observations valid on *t* + *ℓ* for lead ℓ. A single fresh
ERA5-initialized forecast is produced per calendar day (no forecast is carried
across days), so every verified sample is an independent cold-start forecast.

**Rainfall accumulation.** The canonical target is the total precipitation over
the 00–00 UTC calendar day *t* + *ℓ* (mm day⁻¹). Because each model emits
precipitation on its own native cadence, daily totals are formed per model and
then regridded/subset to the common 1° grid:

- **GenCast** emits 12-hourly accumulated precipitation; the daily total is the
  sum of the two 12 h increments covering the valid day (2 × 12 h = 24 h).
- **GraphCast** and **FourCastNet v2** emit 6-hourly accumulations along their
  autoregressive rollout; the daily total is the sum of the four 6 h steps within
  the valid day (4 × 6 h = 24 h).
- **NeuralGCM** (v1_precip) emits a cumulative-mean precipitation *depth*
  accumulated from initialization; the per-day total is recovered by
  differencing consecutive daily values. Validated against CHIRPS on sample
  dates (≈ 2.3 / 4.4 / 5.1 mm day⁻¹ at +1/+2/+3 d for a March 2024 init).

All accumulated totals are clipped at zero and converted to mm day⁻¹.

## Models

We benchmark **four state-of-the-art machine-learning forecast systems** against
a climatological baseline. All learned models are initialized from ERA5, and
their precipitation output is regridded and subset to the common 1° East Africa
domain. To keep verification identical across deterministic and ensemble
systems, every model's output is stored in one canonical layout indexed by
initialization time, ensemble member, lead day, latitude, and longitude, with
precipitation in mm day⁻¹. Deterministic models are stored as single-member
ensembles, so their probabilistic scores reduce to the corresponding
deterministic limits (e.g. CRPS reduces to mean absolute error).

| Model | Family | Native resolution | Native precip step → daily | Members | Precip source |
|---|---|---|---|---|---|
| **GenCast** (Mini) | Diffusion generative ensemble | 1° | 12 h → 2×12 h | 10 | native |
| **GraphCast-small** | Deterministic graph neural network | 1°, 13 levels, mesh 2→5 | 6 h → 4×6 h | 1 (det.) | native |
| **FourCastNet v2** (SFNO) | Spherical Fourier neural operator | 0.25° | 6 h → 4×6 h | 1 (det.) | PrecipAFNO diagnostic |
| **NeuralGCM** (v1_precip) | Hybrid physics-ML | ~2.8° (128×64 Gaussian) | cumulative-mean, de-accumulated | 10 | native (stochastic) |
| **Climatology** | CHIRPS day-of-year distribution | 1° (CHIRPS-derived) | — | 21 | observed |

**GenCast** is a diffusion-based generative ensemble — the first ML model to
outperform the leading operational ensemble (ECMWF ENS) on a majority of
targets — and gives calibrated probabilistic forecasts. We use the public
GenCast-Mini checkpoint (1°), drawing a 10-member ensemble per initialization
(each member from a distinct random seed), which makes it our headline
probabilistic AI baseline for a task where forecast *uncertainty* matters as much
as the central estimate.

**GraphCast** is a deterministic graph neural network run autoregressively that
outperformed the operational deterministic model (HRES) on most variables at a
fraction of the cost, and is the most widely adopted deterministic AI-NWP
baseline. We use the GraphCast-small checkpoint (ERA5 1979–2015, 1°, 13 pressure
levels, mesh 2→5), which gives tractable global rollouts at the benchmark grid.

**FourCastNet v2** is a spherical Fourier neural operator — an architecture
family (operator learning) distinct from GenCast's diffusion model and
GraphCast's message-passing GNN. Including it guards against conclusions that are
specific to one architecture. It does not predict precipitation directly, so we
obtain rainfall via the separate **PrecipitationAFNO** diagnostic (from the
original FourCastNet), which maps the FourCastNet v2 atmospheric state to
6-hourly accumulated precipitation; both run through the NVIDIA earth2mip
framework.

**NeuralGCM** represents the **hybrid physics-ML** paradigm: a differentiable
atmospheric dynamical core coupled to learned sub-grid physics — a fundamentally
different design point from the purely data-driven emulators above. We use the
public `v1_precip` stochastic checkpoint (~2.8°), which predicts precipitation
directly and generates ensemble spread from its *stochastic physics* (a native
source of dispersion rather than input perturbations); we draw 10 members per
initialization. SST and sea-ice forcings over the forecast horizon are prescribed
from ERA5 ("perfect forcing"). Its coarse resolution makes it a useful test of
whether physical constraints compensate for lower spatial detail.

As a reference we include a **climatological baseline** built from the
day-of-year distribution of CHIRPS observations over 2000–2020, giving a
21-member ensemble that is strictly out-of-sample with respect to the 2024
evaluation year and shares the spatial coverage of the verification data. For
precipitation over data-sparse East Africa, beating a well-constructed
day-of-year climatology is a non-trivial bar, which is precisely why it is the
indispensable no-skill reference.

## Reference observations

To ensure our conclusions are not artifacts of a single observational product, we
verify every model against three independent rainfall references, each
conservatively regridded from its native resolution to the common 1° grid using
area-weighted (xESMF) remapping. **CHIRPS v2.0** (0.05°), a blended
gauge–satellite product from the UCSB Climate Hazards Center, is our primary
ground truth. We complement it with **ERA5 total precipitation** (0.25°
reanalysis, hourly accumulations summed to daily totals) and the **TAMSAT v3.1**
satellite rainfall estimate (0.0375°). Reporting skill against all three products
characterizes the sensitivity of our results to observational uncertainty over a
region with sparse gauge coverage.

## Evaluation metrics

We assess both deterministic accuracy and probabilistic calibration over all land
cells, valid dates, and lead times, separately for each observational reference.

For **deterministic accuracy** we report root-mean-square error, mean absolute
error, bias, and spatial correlation of the ensemble mean (or, for deterministic
models, of the single forecast) against observations.

For **probabilistic skill** of the ensemble models, we use the fair continuous
ranked probability score (CRPS) with the Ferro (2014) unbiased estimator, which
removes the finite-ensemble bias of the standard estimator. We characterize
dispersion through ensemble spread and the spread–skill ratio (mean spread ÷
ensemble-mean RMSE), and assess calibration with Talagrand rank histograms
(summarized by a flatness statistic) and with the empirical coverage of the
nominal 50%, 80%, and 90% prediction intervals and their mean widths.

Because operational interest centers on rainfall occurrence at varying intensity,
we evaluate **event-based skill** at exceedance thresholds of 1, 5, 10, and 20 mm
day⁻¹. At each threshold we report the Brier score and Brier skill score, and the
standard contingency-table measures — probability of detection, false-alarm
ratio, critical success index, and frequency bias. We decompose the Brier score
into reliability, resolution, and uncertainty components following Murphy (1973),
and summarize calibration with reliability diagrams.

Finally, to expose **spatial structure** in forecast quality, we map the per-cell
CRPS, CRPS skill score relative to climatology, and bias, and aggregate skill by
country, by latitude band, and as zonal profiles.

## Implementation

The benchmark is a single reproducible pipeline built on xarray, Zarr, xESMF, and
the properscoring/scores libraries. All inputs are public: model checkpoints and
ERA5 initial conditions are read from anonymous Google Cloud Storage, and the
CHIRPS, ERA5, and TAMSAT references are downloaded and cached on first use.
GenCast, GraphCast, and FourCastNet v2 (with PrecipitationAFNO) run in one
JAX + PyTorch/earth2mip environment; **NeuralGCM runs in a dedicated environment**
(its JAX/dinosaur stack is incompatible with the graphcast environment) and is
launched via `run_neuralgcm.sh`. Each model is assigned a separate GPU, and each
initialization is written to its own forecast file (`pred_YYYY-MM-DD.zarr`), so
runs are resumable and reproducible. Inference and verification are separate
stages, so the stored forecasts can be re-scored against any reference or metric
without re-running the models.
