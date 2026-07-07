# Glossary

Short definitions of the metrics and terms used across this site. See
[Methodology](methodology.md#evaluation-metrics) for how each is computed.

## Data products

| Term | Meaning |
|---|---|
| **MAM** | March–April–May — the "long rains" flood season of the Greater Horn of Africa; the 2024 season is the benchmark's evaluation period. |
| **CHIRPS** | Climate Hazards center Infrared Precipitation with Stations — a blended gauge–satellite rainfall product (0.05°); the primary observational reference here. |
| **ERA5** | ECMWF's fifth-generation atmospheric reanalysis; used both as model initial conditions and as a secondary rainfall reference. |
| **TAMSAT** | Tropical Applications of Meteorology using SATellite data — a satellite-only rainfall estimate (0.0375°); a third, independent reference. |
| **Lead day / lead time** | Days between forecast initialization and the valid date being verified (1, 3, 5, 7 in this benchmark). |
| **Init(ialization)** | The date/time a forecast run starts from — MAM 2024 uses 92 daily initializations (1 Mar – 31 May). |

## Deterministic accuracy metrics

| Term | Meaning |
|---|---|
| **Bias** | Mean (forecast − observation); positive = wet bias, negative = dry bias. |
| **MAE** | Mean absolute error — average magnitude of the error, ignoring sign. |
| **RMSE** | Root-mean-square error — like MAE but penalizes large errors more heavily. |
| **ACC** | Anomaly correlation coefficient — spatial correlation between forecast and observed anomalies (departure from climatology) versus lead time. Values above ~0.6 are conventionally considered "useful skill." |

## Probabilistic / ensemble metrics

| Term | Meaning |
|---|---|
| **CRPS** | Continuous Ranked Probability Score — a proper score comparing a full forecast distribution (ensemble) to the observation; generalizes MAE to probabilistic forecasts. Lower is better. This site uses the **fair CRPS** (Ferro 2014), an unbiased estimator for finite ensembles. |
| **CRPSS** | CRPS Skill Score, `1 − CRPS_model / CRPS_reference`. Positive means the model beats the reference (here, climatology); zero is break-even. |
| **Spread** | The ensemble's own standard deviation — how much disagreement exists among members. |
| **Spread–skill ratio (SSR)** | Ensemble spread ÷ ensemble-mean RMSE. SSR = 1 means the ensemble's stated uncertainty matches its actual error (well calibrated); SSR < 1 means the ensemble is **under-dispersive / overconfident**. |
| **Rank (Talagrand) histogram** | Histogram of the rank of the observation within the sorted ensemble members, pooled over many forecasts. Flat = calibrated; U-shaped = under-dispersive; dome-shaped = over-dispersive. |
| **Reliability diagram** | Plots forecast probability (of exceeding a threshold) against the observed frequency of that event; a calibrated forecast lies on the diagonal. |
| **ECE** | Expected Calibration Error — a single number summarizing the average gap between forecast probability and observed frequency in a reliability diagram. Lower is better. |

## Event-based metrics

| Term | Meaning |
|---|---|
| **POD** | Probability of detection — fraction of observed events correctly forecast. Higher is better. |
| **FAR** | False-alarm ratio — fraction of forecast events that did not occur. Lower is better. |
| **CSI** | Critical success index — combines hits, misses and false alarms into one score; balances POD and FAR. Higher is better. |
| **Frequency bias (FB)** | Ratio of forecast to observed event counts. FB = 1 is unbiased; > 1 over-forecasts the event, < 1 under-forecasts it. |
| **Brier score** | Mean squared error of a probability forecast against the binary outcome (0/1). Lower is better; decomposes into reliability, resolution and uncertainty (Murphy 1973). |
| **Brier skill score (BSS)** | Brier score expressed as skill relative to a reference forecast (e.g. climatological event frequency), analogous to CRPSS. |
