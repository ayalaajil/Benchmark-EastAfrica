# East Africa AI Weather Benchmark

A reproducible, probabilistic benchmark of state-of-the-art machine-learning
weather models for **daily rainfall over East Africa**, evaluated against three
independent observational references during the **March–April–May (MAM) 2024
"long rains"** — the primary flood season of the Greater Horn of Africa.

This site is the visual companion to the paper: every headline figure is shown
with a detailed description, alongside the full benchmark pipeline and
methodology, so a reader can grasp the analysis end-to-end without re-running
anything.

## What is benchmarked

| System | Type | Ensemble | Precipitation |
|---|---|---|---|
| **GenCast** (GenCast-Mini) | diffusion generative | 10 members | native, summed 12-hourly |
| **GraphCast** (GraphCast-small) | deterministic GNN | 1 (deterministic) | native, summed 6-hourly |
| **FourCastNet v2** + PrecipitationAFNO | deterministic SFNO | 1 (deterministic) | diagnostic from state |
| **Climatology** | reference baseline | 21 years | CHIRPS day-of-year, 2000–2020 |

All models are initialized from ERA5 (ARCO-ERA5), regridded to a common **1°
East Africa grid** (12°S–15°N, 28°E–52°E), land-masked, and verified at lead
times of **1, 3, 5 and 7 days** against **CHIRPS** (primary), **ERA5** and
**TAMSAT**.

## Headline findings

!!! abstract "At a glance (MAM 2024, vs CHIRPS)"
    - **GenCast is the strongest overall.** It has the lowest CRPS
      (~2.0 mm day⁻¹), the highest anomaly correlation, the best-calibrated
      probabilities, and is the only model that **beats climatology** over the
      equatorial belt.
    - **All models have limited deterministic skill.** Anomaly correlation stays
      **below the 0.6 "useful-skill" guide at every lead**, and degrades with
      lead time — these are genuinely hard, convection-dominated rains.
    - **Distinct bias signatures.** FourCastNet is systematically **dry**
      (≈ −1.2 mm day⁻¹); GraphCast drifts **wet** at longer leads (+0.84 mm day⁻¹
      by day 7); GenCast is nearly **unbiased** in the domain mean.
    - **GenCast is under-dispersive (overconfident).** Spread-skill ratio rises
      from ~0.24 (day 1) toward ~0.45, and rank histograms are U-shaped — the
      ensemble is too narrow, especially at short lead.
    - **Results are observation-sensitive.** Skill and calibration shift visibly
      across CHIRPS / ERA5 / TAMSAT, underscoring observational uncertainty over
      this data-sparse region.

## How to read this site

- **[Pipeline](pipeline.md)** — the end-to-end system: a single configurable
  inference runner, the four model adapters, and the verification stage.
- **[Experimental Setup](experimental-setup.md)** — domain, models, reference
  products, and the full metric catalogue.
- **Results** — organized by metric family:
    - **[Deterministic Skill](results/deterministic-skill.md)** — time series,
      bias/MAE, spatial and zonal error structure, anomaly correlation.
    - **[Probabilistic Calibration](results/probabilistic-calibration.md)** —
      CRPS, spread–skill, rank histograms, reliability.
    - **[Skill vs Climatology](results/skill-vs-climatology.md)** — the CRPS
      skill score maps.
    - **[Event-Based Skill](results/event-based.md)** — exceedance thresholds,
      Brier and contingency scores.
- **[Reproducibility](reproducibility.md)** — exact commands to regenerate every
  prediction, table and figure.

!!! note "Evaluation period"
    Figures on this site show **MAM 2024** (92 daily initializations, 1 Mar –
    31 May, verified through 7 Jun). The pipeline also supports full-year and
    all-variable runs; figures are regenerated from the same scripts.
