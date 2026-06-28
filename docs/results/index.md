# Results — Overview

All figures show **MAM 2024** (92 daily initializations, verified at leads 1, 3,
5, 7 days), with **CHIRPS** as the primary reference unless stated. Results
against ERA5 and TAMSAT are summarized in the tables and discussed where they
change the picture.

## Domain-mean error summary

Ensemble-mean error over all land cells and valid dates, lead day 1:

| Model | vs CHIRPS — Bias | MAE | RMSE | vs ERA5 — Bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| **FourCastNet** | −1.18 | 2.66 | 6.01 | −1.34 | 2.59 | 6.78 |
| **GenCast** | −0.19 | 2.82 | 6.81 | −0.11 | 2.13 | 5.73 |
| **GraphCast** | −0.14 | 3.15 | 6.97 | +0.05 | 3.23 | 8.08 |

_(mm day⁻¹.)_ FourCastNet is the **driest** (large negative bias); GenCast is
nearly unbiased and has the lowest MAE against ERA5; GraphCast has the largest
MAE/RMSE.

## Skill vs lead time (vs CHIRPS)

Spatial correlation of the ensemble mean, and GenCast CRPS:

| Lead | FourCastNet corr | GenCast corr | GraphCast corr | GenCast CRPS (mm day⁻¹) |
|---:|---:|---:|---:|---:|
| 1 | 0.41 | **0.43** | 0.36 | 2.25 |
| 3 | 0.40 | **0.49** | 0.31 | 2.02 |
| 5 | 0.40 | **0.46** | 0.26 | 2.00 |
| 7 | 0.34 | **0.44** | 0.19 | 2.01 |

GenCast leads at every horizon. GraphCast degrades fastest with lead time;
GenCast's CRPS is remarkably flat. GenCast's ensemble is **under-dispersive
(overconfident)** — quantified on the
[Probabilistic Calibration](probabilistic-calibration.md) page.

## Reading guide

| Page | Question it answers | Figures |
|---|---|---|
| [Deterministic Skill](deterministic-skill.md) | How accurate are the forecasts, and where do errors live? | time series, bias/MAE, spatial maps, zonal profiles, anomaly correlation |
| [Probabilistic Calibration](probabilistic-calibration.md) | Is GenCast's uncertainty trustworthy? | CRPS/spread, spread–skill, rank histograms, reliability |
| [Skill vs Climatology](skill-vs-climatology.md) | Do the models beat a climatological baseline? | CRPS skill score maps |
| [Event-Based Skill](event-based.md) | How well are rainfall thresholds detected? | Brier & contingency scores |

!!! note
    Figures are produced by `run_verification.py` and the companion analysis
    scripts; the CSV tables behind them live in `mam2024_analysis_outputs/`. See
    [Reproducibility](../reproducibility.md).
