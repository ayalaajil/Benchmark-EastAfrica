# East Africa AI Weather Benchmark

A reproducible, probabilistic benchmark of state-of-the-art machine-learning
weather models for **daily rainfall over East Africa**, evaluated against three
independent observational references during the **March–April–May (MAM) 2024
"long rains"** — the primary flood season of the Greater Horn of Africa.

This site is the visual companion to the paper: every headline figure is shown
with a detailed description, alongside the full benchmark pipeline and
methodology, so a reader can grasp the analysis end-to-end without re-running
anything.

## Scorecard (MAM 2024, vs CHIRPS, lead day 1 unless noted)

| Model | Type | CRPS / MAE ↓ (mm day⁻¹) | ACC ↑ (day 1 → 7) | Bias signature | Event CSI ↑ | Beats climatology? |
|---|---|---:|---|---|---:|---|
| **GenCast** | ensemble (diffusion) | **2.25** (fair CRPS; flattest with lead) | **0.43 → 0.44** (best, most stable) | ≈ unbiased (−0.19) | **0.54** | **Yes** — equatorial belt, holds to day 7 |
| FourCastNet | deterministic | 2.66 (MAE) | 0.41 → 0.34 | Dry (−1.18) | 0.49 | Mixed — negative over northern Horn |
| GraphCast | deterministic | 3.15 (MAE) | 0.36 → 0.19 (fastest decay) | Wet drift (−0.14 → +0.84) | 0.51 | No — mostly worse, worsens with lead |
| Climatology | 21-yr baseline | reference | — | — | — | reference |

**Verdict:** GenCast is the strongest system on every axis, but its ensemble is
overconfident (see [finding 4](key-findings.md#4-gencast-is-under-dispersive-overconfident)),
and no model reliably beats climatology outside the equatorial belt. Full
numbers and derivations: [Key Findings](key-findings.md) ·
[Results](results/index.md).

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

!!! abstract "At a glance — see [Key Findings](key-findings.md) for the full write-up"
    1. **GenCast is the strongest overall**, and the only model that beats climatology.
    2. **All models have limited deterministic skill** — ACC stays below the 0.6 useful-skill guide.
    3. **Distinct bias signatures** — FourCastNet dry, GraphCast increasingly wet, GenCast near-unbiased.
    4. **GenCast is under-dispersive (overconfident)** — its ensemble is too narrow for its error.
    5. **Results are observation-sensitive** — skill and calibration shift across CHIRPS / ERA5 / TAMSAT.

## How to read this site

- **[Key Findings](key-findings.md)** — the five headline results, each with
  its supporting figure.
- **[Methodology](methodology.md)** — the end-to-end pipeline (inference +
  verification) and the full experimental setup: domain, models, reference
  products, metric catalogue.
- **Results** — organized by metric family:
    - **[Overview](results/index.md)** — domain-mean summary tables.
    - **[Deterministic Skill](results/deterministic-skill.md)** — time series,
      bias/MAE, spatial and zonal error structure, anomaly correlation.
    - **[Probabilistic Calibration](results/probabilistic-calibration.md)** —
      CRPS, spread–skill, rank histograms, reliability.
    - **[Skill vs Climatology](results/skill-vs-climatology.md)** — the CRPS
      skill score maps.
    - **[Event-Based Skill](results/event-based.md)** — exceedance thresholds,
      Brier and contingency scores.
- **[Glossary](glossary.md)** — definitions of every metric used on this site.
- **[Data & Downloads](data.md)** — figures, CSV tables, and how to regenerate them.
- **[Reproducibility](reproducibility.md)** — exact commands to regenerate every
  prediction, table and figure.

!!! note "Evaluation period"
    Figures on this site show **MAM 2024** (92 daily initializations, 1 Mar –
    31 May, verified through 7 Jun). The pipeline also supports full-year and
    all-variable runs; figures are regenerated from the same scripts.
