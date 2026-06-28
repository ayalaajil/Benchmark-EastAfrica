# Event-Based Skill

Operational interest centers on **whether it rains** at a given intensity, not
just the amount. We evaluate forecasts as binary events at exceedance thresholds
of **1, 5, 10 and 20 mm day⁻¹** using contingency-table scores and, for GenCast,
the Brier score.

The figures on this page are tabulated rather than plotted; full tables live in
`event_scores_by_threshold.csv` and `gencast_brier_scores.csv`.

## Contingency scores at the "rain / no-rain" threshold

Detection skill for **≥ 1 mm day⁻¹** events, lead day 1, vs CHIRPS
(POD = probability of detection, FAR = false-alarm ratio, CSI = critical success
index, FB = frequency bias; **FB = 1 is unbiased event frequency**):

| Model | POD ↑ | FAR ↓ | CSI ↑ | Frequency bias |
|---|---:|---:|---:|---:|
| FourCastNet | 0.62 | 0.30 | 0.49 | **0.88** (under-forecasts) |
| **GenCast** | 0.70 | **0.29** | **0.54** | **0.99** (≈ unbiased) |
| GraphCast | **0.73** | 0.37 | 0.51 | **1.15** (over-forecasts) |

- **GenCast has the best balance** — the highest CSI and a frequency bias of
  ~1.0, i.e. it predicts rain about as often as it is observed.
- **GraphCast detects the most events (highest POD) but over-forecasts** (FB 1.15,
  higher FAR) — it rains too readily.
- **FourCastNet under-forecasts** rain occurrence (FB 0.88), consistent with its
  dry amount bias.

## Behaviour with intensity

As the threshold rises from 1 → 20 mm day⁻¹, events become rarer and **all skill
scores fall** — POD and CSI decrease while FAR climbs (e.g. GenCast CSI drops
from 0.54 at 1 mm to ~0.23 at 10 mm and ~0.12 at 20 mm). Heavy, localized
convective rainfall is the hardest to place correctly. The models keep their
character across thresholds: GraphCast remains the most over-forecasting,
FourCastNet the most conservative, GenCast the most balanced.

## GenCast Brier score

Brier score of GenCast's exceedance probabilities vs CHIRPS, lead day 1
(lower is better; the base rate gives context):

| Threshold | Brier score | Observed event rate |
|---|---:|---:|
| ≥ 1 mm | 0.179 | 0.369 |
| ≥ 5 mm | 0.131 | 0.186 |
| ≥ 10 mm | 0.084 | 0.092 |
| ≥ 20 mm | 0.029 | 0.026 |

The Brier score is **nearly flat across lead time** (it changes by < 0.01 from
lead 1 to lead 7), mirroring GenCast's stable CRPS. Reliability-diagram and ECE
diagnostics for these same probabilities are on the
[Probabilistic Calibration](probabilistic-calibration.md) page; the Brier-score
reliability/resolution/uncertainty decomposition is tabulated in
`gencast_brier_scores.csv` / `gencast_reliability_tables.csv`.
