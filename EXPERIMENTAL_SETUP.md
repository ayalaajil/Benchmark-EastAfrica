# Experimental Setup

## Study domain and evaluation period

We evaluate daily rainfall forecasts over East Africa on a common regular grid
spanning 12°S–15°N and 28°E–52°E at 1° resolution (28 × 25 = 700 grid cells),
which matches the native output resolution of the graph-based models considered.
Verification is restricted to land using a mask derived from the Natural Earth
50 m land polygons (approximately 512 cells); ocean and large inland-lake cells
are excluded throughout. For regional analysis we further subset the domain into
the seven flood-season countries of the Greater Horn of Africa — Kenya,
Ethiopia, Tanzania, Somalia, Uganda, Rwanda, and Burundi — and into zonal
latitude bands. All spatially aggregated statistics are computed with
cos(latitude) area weights so that regional means are not biased toward the
higher latitudes of the domain.

Our experiments target the March–April–May (MAM) 2024 "long rains," the primary
rainy season over the region. We initialize each model daily over the full
season, yielding 92 initialization dates from 1 March to 31 May 2024, and verify
forecasts at lead times of 1, 3, 5, and 7 days. Each forecast initialized at
00 UTC on date *t* is compared against observations valid on *t* + *ℓ* for lead
time *ℓ*, so that the verified valid dates extend through 7 June 2024.

## Models

We benchmark three state-of-the-art machine-learning forecast systems against a
climatological baseline. All learned models are initialized from ERA5 analyses
(drawn from the public ARCO-ERA5 archive) and their precipitation output is
regridded and subset to the common 1° East Africa domain. To keep verification
identical across deterministic and ensemble systems, every model's output is
stored in a canonical layout indexed by initialization time, ensemble member,
lead day, latitude, and longitude, with precipitation expressed in mm day⁻¹.

**GenCast** is a diffusion-based generative ensemble model; we use the public
GenCast-Mini checkpoint and draw a 10-member ensemble per initialization,
summing its native 12-hourly increments to daily totals. **GraphCast** is a
deterministic graph neural network run autoregressively; we use the
GraphCast-small checkpoint trained on ERA5 (1979–2015) at 1° resolution with 13
pressure levels. **FourCastNet v2** is a deterministic spherical-Fourier neural
operator that does not predict precipitation directly; we obtain rainfall via
the separate PrecipitationAFNO diagnostic, which maps the FourCastNet v2 state
to 6-hourly accumulated precipitation, and run both through the NVIDIA earth2mip
framework. Since GraphCast and FourCastNet v2 are deterministic, they are stored
as single-member ensembles, and their probabilistic scores reduce to the
corresponding deterministic limits (for example, the continuous ranked
probability score reduces to mean absolute error).

As a reference, we include a **climatological baseline** built from the
day-of-year distribution of CHIRPS observations over the reference period
2000–2020, giving a 21-member ensemble that is strictly out-of-sample with
respect to the 2024 evaluation year and shares the spatial coverage of the
verification data.

## Reference observations

To ensure that our conclusions are not artifacts of a single observational
product, we verify every model against three independent rainfall references,
each conservatively regridded from its native resolution to the common 1° grid
using area-weighted (xESMF) remapping. CHIRPS v2.0 (0.05°), a blended
gauge–satellite product from the UCSB Climate Hazards Center, serves as our
primary ground truth. We complement it with ERA5 total precipitation (0.25°
reanalysis, with hourly accumulations summed to daily totals) and the TAMSAT
v3.1 satellite rainfall estimate (0.0375°). Reporting skill against all three
products characterizes the sensitivity of our results to observational
uncertainty over a data-sparse region.

## Evaluation metrics

We assess both deterministic accuracy and probabilistic calibration over all
land cells, valid dates, and lead times, separately for each observational
reference.

For deterministic accuracy we report the root-mean-square error, mean absolute
error, bias, and spatial correlation of the ensemble mean (or, for deterministic
models, of the single forecast) against observations.

For probabilistic skill of the ensemble models, we use the fair continuous
ranked probability score (CRPS) computed with the Ferro (2014) unbiased
estimator, which removes the finite-ensemble bias of the standard estimator.
We characterize ensemble dispersion through the ensemble spread and the
spread–skill ratio (mean spread divided by ensemble-mean RMSE), and assess
ensemble calibration with Talagrand rank histograms, summarized by a flatness
statistic, and with the empirical coverage of the nominal 50%, 80%, and 90%
prediction intervals together with their mean widths.

Because operational interest centers on the occurrence of rainfall of varying
intensity, we additionally evaluate event-based skill at exceedance thresholds
of 1, 5, 10, and 20 mm day⁻¹. At each threshold we report the Brier score and
Brier skill score, and the standard contingency-table measures — probability of
detection, false-alarm ratio, critical success index, and frequency bias. We
decompose the Brier score into its reliability, resolution, and uncertainty
components following Murphy (1973), and summarize calibration with reliability
diagrams and the expected calibration error.

Finally, to expose spatial structure in forecast quality, we map the per-cell
CRPS, CRPS skill score relative to climatology, and bias, and aggregate skill by
country, by latitude band, and as zonal profiles.

## Implementation

The benchmark is implemented as a single reproducible pipeline built on xarray,
Zarr, xESMF, and the properscoring/scores libraries. All inputs are publicly
available: model checkpoints and ERA5 initial conditions are read from anonymous
Google Cloud Storage, and the CHIRPS, ERA5, and TAMSAT references are downloaded
and cached on first use. Inference for all four models runs within a single
software environment — JAX with the graphcast package for GenCast and GraphCast,
and PyTorch with earth2mip for FourCastNet v2 and PrecipitationAFNO — with each
model assigned to a separate GPU and each initialization written to its own
forecast file, so that runs are resumable and reproducible.
