# Freight Rate Prediction

A production machine learning system that prices freight loads from their lane,
distance, trailer type, weight, and date. Freight brokers normally quote rates by
instinct, which does not scale, varies between people, and cannot look forward.
This replaces that judgement with a model, and wraps it in the serving,
monitoring, and retraining infrastructure a real deployment needs.

The problem is harder than it looks. The labelled data ends on 31 October and
every load that needs a price falls in November or December, so every prediction
is a forecast into months the model has never seen.

---

## Results

Scored on a time-based holdout. Trained on January to August, tested on September
and October, months the model never saw during training.

| Model | RMSE | MAE | MAPE | R² | Bias |
|---|---|---|---|---|---|
| Baseline (median rate per mile) | $112.61 | $83.04 | 3.85% | 0.9933 | −$12.19 |
| **LightGBM** | **$67.65** | **$45.11** | **1.88%** | **0.9976** | **$0.10** |

**39.9% better than the baseline.** Across three rolling-origin folds:
RMSE 88.91 (± 42.16).

The near-zero bias matters as much as the RMSE. A model forecasting two months
ahead can easily drift high or low, and a consistent lean would cost money on
every load quoted. This one is effectively unbiased.

**Live behaviour.** Replaying all 12,000 held-out loads through the API, error
holds steady at 3.35% through November then climbs to 4.61% by 31 December, as
the model under-quotes into a peak season it was never trained on. The drift
monitor flags 12.1% of traffic on lanes with no training history, matching the
count of unseen cities exactly.

---

## Features

**Modelling**

- Gradient-boosted regression on `log(rate per mile)`, with a smearing correction
  when converting back to dollars
- Fourier seasonal terms and a fitted annual curve, so the calendar stays
  meaningful past the end of the training data
- Coordinate-based geography that prices cities the model has never seen
- Time-based splitting with rolling-origin cross-validation and an automatic
  leakage guard

**Serving**

- FastAPI service reusing the offline pipeline, verified to produce identical
  predictions to the batch path
- Single and batch endpoints, with per-prediction warnings for unfamiliar cities,
  dates beyond training, and imputed values
- Sub-millisecond inference at batch size 1,000

**Monitoring**

- Prediction logging to Postgres, joined to outcomes as they settle
- Delayed-feedback accuracy that reports coverage alongside every metric
- Drift detection by Population Stability Index and by Evidently
- Live Dash dashboard with five tabs, updating without a page refresh

**Operations**

- MLflow experiment tracking and model registry
- Prefect flows for scheduled monitoring and retraining
- A promotion gate that refuses to ship a model which does not beat production
- GitHub Actions for tests, model checks, and Docker builds
- 91 automated tests, mutation-tested

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | Python 3.12, LightGBM, scikit-learn, pandas, NumPy, SciPy |
| Serving | FastAPI, Uvicorn, Pydantic |
| Storage | PostgreSQL, SQLAlchemy Core, SQLite (local fallback) |
| Monitoring | Evidently, Dash, Plotly |
| Tracking | MLflow |
| Orchestration | Prefect |
| Infrastructure | Docker, Docker Compose, GitHub Actions |
| Quality | pytest, ruff |

---

## Architecture

![System architecture](reports/figures/architecture.png)

Training fits every transform once and persists it. Inference reloads those exact
transforms and refits nothing. The imputation medians, city coordinates, category
codes, and seasonal curve all come from the training run.

That single rule is why training and serving are separate modules sharing one
bundle rather than one pipeline with a flag. Recomputing anything on the data
being scored would leak information backwards and quietly break the parity
between an offline prediction and a served one.

---

## Project Structure

```
freight-rate-prediction/
├── config/
│   ├── config.yaml              every setting, with the reasoning beside it
│   └── logging.yaml             console and rotating file logging
├── data/                        the provided CSVs and pipeline artifacts
├── db/
│   ├── 00-databases.sql         creates the MLflow database
│   └── 01-schema.sql            predictions, actuals, and the scored view
├── docker/
│   ├── Dockerfile               API image
│   ├── Dockerfile.dashboard     dashboard image
│   └── Dockerfile.mlflow        tracking server image
├── entrypoint/
│   ├── train.py                 CLI: trains and saves the model
│   └── predict.py               CLI: writes both output files
├── notebooks/
│   ├── 01_eda.ipynb             what drives the price
│   └── 02_data_quality.ipynb    every defect, investigated then fixed
├── src/
│   ├── config.py                typed, validated configuration
│   ├── logger.py                logging setup and helpers
│   ├── data/                    loading, cleaning
│   ├── features/                seasonality, geography, encoding, assembly
│   ├── models/                  baseline, estimator, persistence
│   ├── validation/              splitters, metrics
│   └── pipelines/               training and inference
├── serving/                     FastAPI app, schemas, prediction store
├── monitoring/                  performance metrics and drift detection
├── dashboard/                   Dash app, charts, and five tabs
├── tracking/                    MLflow integration
├── flows/                       Prefect flows and the promotion gate
├── simulator/                   replays the validation set as live traffic
├── tests/                       91 tests
├── .github/workflows/           tests, model check, Docker build
├── models/                      trained bundle and metadata
├── reports/figures/             EDA, cleaning, and architecture figures
├── score.py                     provided, unmodified
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Installation and Setup

Requires **Python 3.12** (3.11 also works). Docker is optional but recommended
for the full stack.

```bash
git clone <repository-url>
cd freight-rate-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the provided CSVs are in `data/`: `train_test.csv`, `validation.csv`, and
`december_chart_inputs.csv`.

---

## Usage

### Train and predict

```bash
python entrypoint/train.py
python entrypoint/predict.py --score
```

About 90 seconds end to end. Produces:

```
validation_predictions.csv               the submission, 12,000 rows
data/december_chart_inputs.csv           predicted_rate filled in place
scorer_results/candidate_december.png    the December chart
models/                                  trained bundle and metadata
```

### Run the full stack

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Dashboard | http://localhost:8050 |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

Train before building. Both the API and dashboard images read `models/`, so an
untrained repository gives you services with nothing to serve.

### Generate live traffic

```bash
export DATABASE_URL=postgresql+psycopg2://freight:freight@localhost:5432/freight_monitoring
python simulator/replay.py --speed 200
```

Streams the 12,000 held-out loads through the API day by day, reporting synthetic
outcomes on a delay so the feedback lag is real. Watch the dashboard while it
runs. Options: `--scenario peak_season|shock|baseline`, `--days N`,
`--feedback-delay N`.

### Orchestration

```bash
python flows/monitor.py --days 7      # check drift and accuracy once
python flows/retrain.py --dry-run     # train and compare without promoting
python flows/retrain.py               # promote only if it beats production
python flows/deployments.py           # register both flows on their schedules
```

### Make targets

```bash
make help          # every target
make all           # install, train, predict, score
make test          # the test suite
make up            # build and start the full stack
make ci            # everything the CI pipeline runs, locally
```

`make` is not available on Windows by default. The `python` commands above do
exactly the same thing.

### Tests

```bash
pytest                       # 91 tests
pytest -m "not integration"  # the subset needing no data files
```

---

## Screenshots

### December prediction chart

![December chart](scorer_results/candidate_december.png)

Produced by the supplied scorer. One load held constant across every day of
December, with only the date varying. All 31 days differ, confirming the seasonal
features still work past the end of the training data.

### Monitoring dashboard

![Dashboard overview](reports/figures/overview1.png)
![Model Drift tab](reports/figures/model_drift.png)
![Data Drift tab](reports/figures/data_drift.png)

The overview tracks predicted against actual rates as they settle. The drift tab
compares live traffic against the training data, splitting production into the
part that still resembles training and the part that does not.

---

## Engineering Decisions

Three modelling choices were made by running the alternatives and comparing on
the folds, not by argument. Each is recorded in `config/config.yaml` beside the
setting it controls.

**Seasonal curve as an offset, or as a feature?** Forcing the fitted curve
through as an offset guarantees full strength. It also lost on every fold, mean
RMSE 108.02 against 88.91. The capability remains behind
`model.seasonal_offset`, switched off.

**Day-of-week features?** The weekly spread is only 2%. Removing them improved
mean RMSE from 91.87 to 88.91 and won on two folds of three, so they are out.

**How much seasonality to force?** A direct trade-off between the accuracy score
and how much the December curve moves:

| Configuration | Mean RMSE | Dec trend | Dec spread |
|---|---|---|---|
| Weekday + Fourier + curve | 91.87 | +$5.45 | $21.67 |
| **Fourier + curve** | **88.91** | +$0.82 | $9.77 |
| Curve only | 111.48 | −$36.87 | $36.97 |

The middle row is the default, because accuracy is what is scored. The
consequence is a December chart varying by 1.23% rather than showing a pronounced
trend, which is an honest reflection of what the data supports.

Four architectural decisions shaped the rest.

**Row removal is training-only.** Validation carries the same defects as
training, but the scorer demands a rate for all 12,000 loads. Corrupted rows are
dropped while learning and repaired while predicting. This is the `is_training`
flag in `src/data/cleaning.py`, and it is directly tested.

**Logging never breaks a prediction.** Writes go through background tasks after
the response is sent, and a database failure returns zero rather than raising.
Reads deliberately do the opposite and fail loudly, because a caller asking for
metrics needs to know the answer is missing rather than receiving an empty result
that looks like data.

**No raw month or day-of-year column reaches the model.** Training never sees
months 11 or 12, so a tree splitting on them would return the October answer for
every prediction and flatten the December chart. The calendar arrives only
through bounded Fourier terms and a fitted curve, both defined for any date.

**A candidate model has to earn promotion.** Scheduled retraining that ships
whatever it produced is worse than no retraining, because a bad month of data
degrades the service silently. Every candidate is trained into a staging
directory and must beat production by at least 1%, beat the baseline, and sit
inside hard RMSE and MAPE ceilings before it replaces anything.

---

## Limitations and Future Improvements

**December seasonality is the open question.** The ten months of labelled data
contain a clear annual cycle and no holiday effects at all, tested against four
public holidays. That cycle is projected forward on the assumption it is
periodic, which is reasonable but unverifiable from the data given. If real
December rates spike for peak season, these predictions run low. This is the
largest single source of uncertainty in the system.

**The model needs history.** With only six months of training data the error more
than doubles, as the earliest cross-validation fold shows. The approach depends
on seeing enough of the annual shape, which is an argument for retraining monthly
rather than annually.

**Unseen cities are handled but not solved.** Coordinates let the model place
Chicago sensibly despite never having priced a load there. That is far better
than crashing, but a load out of a major market with no history is still an
educated guess.

**Replay outcomes are synthetic.** The validation set has no rates, so the
simulator manufactures them. This is stated wherever it surfaces, and the default
scenario is designed to make the model look worse rather than better.

**Planned next**

- Quantile models to return a price range rather than a single number, since a
  broker quoting a rate cares about the downside as much as the middle
- Regional grouping so thin lanes have something more reliable than their own
  twelve observations
- A proper hyperparameter search, not yet run because the gain over sensible
  defaults looked small next to the decisions above
- External freight market data to anchor the seasonal projection beyond the ten
  months available
- Automated rollback when live metrics degrade after a promotion