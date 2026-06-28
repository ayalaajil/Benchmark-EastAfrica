# Experimental Setup

## Study domain and evaluation period

We evaluate daily rainfall forecasts over East Africa on a common regular grid
spanning **12°S–15°N and 28°E–52°E at 1° resolution** (28 × 25 = 700 grid
cells), matching the native output resolution of the graph-based models.
Verification is restricted to **land** using a mask derived from the Natural
Earth 50 m land polygons (~512 cells); ocean and large inland-lake cells are
excluded throughout.

For regional analysis the domain is further subset into the seven flood-season
countries of the Greater Horn of Africa — **Kenya, Ethiopia, Tanzania, Somalia,
Uganda, Rwanda and Burundi** — and into zonal latitude bands. All spatially
aggregated statistics use **cos(latitude) area weights** so regional means are
not biased toward higher latitudes.

The experiments target the **March–April–May (MAM) 2024 long rains**. Each model
is initialized daily over the full season — **92 initializations from 1 March to
31 May 2024** — and verified at lead times of **1, 3, 5 and 7 days**. A forecast
initialized at 00 UTC on date *t* is compared against observations valid on
*t + ℓ*, so verified valid dates extend through 7 June 2024.

## Models

Three state-of-the-art ML forecast systems are benchmarked against a
climatological baseline. All learned models are initialized from ERA5 analyses
(public ARCO-ERA5 archive); their precipitation is regridded and subset to the
common 1° domain and expressed in mm day⁻¹.

- **GenCast** — diffusion-based generative ensemble (GenCast-Mini checkpoint),
  10 members per initialization, native 12-hourly increments summed to daily.
- **GraphCast** — deterministic graph neural network (GraphCast-small, ERA5
  1979–2015, 1°, 13 pressure levels), run autoregressively.
- **FourCastNet v2** — deterministic spherical-Fourier neural operator; rainfall
  via the separate PrecipitationAFNO diagnostic (6-hourly accumulation), run
  through NVIDIA earth2mip.

GraphCast and FourCastNet are deterministic, stored as single-member ensembles;
their probabilistic scores reduce to the corresponding deterministic limits
(e.g. CRPS → MAE).

The **climatological baseline** is built from the day-of-year distribution of
CHIRPS over the reference period **2000–2020**, giving a 21-member ensemble that
is strictly out-of-sample with respect to 2024.

## Reference observations

To ensure conclusions are not artifacts of a single product, every model is
verified against **three independent rainfall references**, each conservatively
regridded (area-weighted xESMF) to the common 1° grid:

| Product | Native res. | Type |
|---|---|---|
| **CHIRPS v2.0** (primary) | 0.05° | blended gauge–satellite (UCSB) |
| **ERA5** total precipitation | 0.25° | reanalysis (hourly → daily) |
| **TAMSAT v3.1** | 0.0375° | satellite estimate |

Reporting skill against all three characterizes sensitivity to observational
uncertainty over this data-sparse region.

## Evaluation metrics

Both deterministic accuracy and probabilistic calibration are assessed over all
land cells, valid dates and lead times, separately for each reference.

**Deterministic accuracy** — RMSE, MAE, bias, and spatial correlation of the
ensemble mean (or single forecast) against observations; plus **anomaly
correlation** versus lead.

**Probabilistic skill** — the **fair CRPS** (Ferro 2014 unbiased estimator),
ensemble **spread** and **spread–skill ratio** (mean spread / ensemble-mean
RMSE), **Talagrand rank histograms**, and the empirical coverage and width of
nominal 50/80/90 % prediction intervals.

**Event-based skill** — at exceedance thresholds of **1, 5, 10 and 20 mm day⁻¹**:
Brier score and Brier skill score; probability of detection, false-alarm ratio,
critical success index, and frequency bias; Brier-score decomposition
(reliability/resolution/uncertainty, Murphy 1973); reliability diagrams and
expected calibration error (ECE).

**Skill vs climatology** — the per-cell **CRPS skill score**,
`CRPSS = 1 − CRPS_model / CRPS_climatology`, referenced to the out-of-sample
CHIRPS day-of-year climatology.

**Spatial structure** — per-cell CRPS, CRPSS and bias maps, aggregated by
country, by latitude band, and as zonal profiles.

## Implementation

All inputs are public: model checkpoints and ERA5 initial conditions are read
from anonymous Google Cloud Storage; CHIRPS, ERA5 and TAMSAT references are
downloaded and cached on first use. Inference for all models runs within a
single software environment — JAX + graphcast for GenCast/GraphCast, PyTorch +
earth2mip for FourCastNet/PrecipitationAFNO — with each initialization written
to its own resumable forecast file. See **[Reproducibility](reproducibility.md)**.
