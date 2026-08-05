# Freight Rate Prediction

Predicting the dollar rate for a freight load from its lane, distance, equipment,
weight, and date.

The labelled data covers **1 January to 31 October 2025**. Every load that needs a
prediction falls in **November and December**, with no overlap in time. That makes
this a forecasting problem rather than a gap-filling one, and it shapes almost
every decision below.

## Results

Scored on a time-based holdout. Trained on January to August, tested on
September and October, months the model had never seen.

| Model | RMSE | MAE | MAPE | R² | Bias |
|---|---|---|---|---|---|
| Baseline (median rate per mile) | $112.61 | $83.04 | 3.85% | 0.9933 | −$12.19 |
| **LightGBM** | **$67.65** | **$45.11** | **1.88%** | **0.9976** | **$0.10** |

**39.9% better than the baseline.** Across three rolling-origin folds:
RMSE 88.91 (± 42.16).

The near-zero bias matters as much as the RMSE. A model predicting two months
forward can easily drift high or low, and this one does not.

## Architecture

![System architecture](reports/figures/architecture.png)

Training fits every transform once and persists it. Inference reloads those exact
transforms and refits nothing. The imputation medians, city coordinates, category
codes, and seasonal curve all come from the training run. That single rule is why
the two pipelines are separate modules sharing one bundle rather than one pipeline
with a flag.

## Quick start

Requires **Python 3.12** (3.11 also works).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python entrypoint/train.py
python entrypoint/predict.py --score
```

Takes a few seconds(about 90 seconds) end to end. That produces everything:

```
validation_predictions.csv               the submission, 12,000 rows
data/december_chart_inputs.csv           predicted_rate filled in place
scorer_results/candidate_december.png    the chart for the report
models/                                  trained model and its metadata
```

### With make

```bash
make all        # install, train, predict, score
make help       # every target
```

`make` is not available on Windows by default. Use the two `python` commands
above instead, they do exactly the same thing.

### Running the scorer on its own

The command from the assessment instructions, which works from the project root:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`predict.py --score` already runs this. Use it directly to redraw the chart
without regenerating predictions.

## Approach

### Data quality

Five defects were found and fixed. Full working in
`notebooks/02_data_quality.ipynb`.

| Issue | Train / Validation | What it is | Treatment |
|---|---|---|---|
| Negative weight | 292 / 145 | A sign flip. Bounds and distribution match the valid weights exactly, to the pound | `abs()`, recovering the true value |
| Missing weight | 300 / 165 | Missing at random | Median per equipment type |
| Missing `market_index` | 374 / 249 | Missing at random | Median; the feature is excluded anyway |
| Inflated rates | 340 / n/a | About 3.5× the baseline per-mile rate, up to $14.13/mile | Dropped from training only |
| Deflated rates | 329 / n/a | About 0.28× the baseline, down to $0.33/mile | Dropped from training only |

Rates are judged **per mile**, not in dollars. A $12,000 rate is normal over
3,400 miles and absurd over 200, so raw dollars cannot separate a corrupt record
from a long haul.

Cleaning removed 669 training rows (1.39%) and left the standard deviation of
rate per mile 53% lower with the mean and median unchanged. noise removed, not
signal.

Two rules govern the code:

1. **Row removal is training-only.** Validation carries the same defects, but the
   scorer demands a rate for all 12,000 loads, so at prediction time every defect
   is repaired rather than dropped. This is the `is_training` flag in
   `src/data/cleaning.py`.
2. **Imputation values are learned on training data and reused.** Recomputing
   them on the data being scored would leak information back into the pipeline.
   They travel with the model as `models/cleaning_artifacts.json`.

### Features

24 features in five groups. `src/features/`.

| Group | Features |
|---|---|
| Geographic | distance, log distance, four coordinates, great-circle distance, circuity, lane direction, midpoint |
| Load | weight |
| Categorical | pickup, delivery, equipment codes, unknown-city flag |
| Calendar | six Fourier terms |
| Seasonal | fitted annual curve |

Three decisions worth explaining.

**Coordinates are kept, and they matter.** Eight cities appear only in the
validation set, including Chicago and Charlotte, covering **12% of the loads to
be priced**. Encoding cities by name alone would break on all of them. Latitude
and longitude are continuous, so an unfamiliar city still has a usable position.
Unknown names map to a reserved code and set a flag the model can split on.

**No raw month or day-of-year column reaches the model.** Training never sees
months 11 or 12, so a tree splitting on them would reuse the October answer for
every prediction. The calendar reaches the model only through bounded Fourier
terms and a fitted seasonal curve, both of which are defined in December.

**`market_index` and `quote_signal` are excluded.** Neither is a market-level
indicator despite the names. Both vary hundreds of times within a single date.
They correlate with rate per mile at 0.18 and 0.10, against distance at 0.91. And
neither exists in `december_chart_inputs.csv`. Dropping them costs almost nothing
and means one model serves both output files.

### Validation and split

Validation runs 1 November to 31 December; training ends 31 October. **Zero
overlap.** A random split would let October teach the model about September and
report a score that would not survive the real task.

```
Holdout:  train Jan 01 – Aug 31 (37,951)   test Sep 01 – Oct 31 (9,380)
fold_1:   train Jan 01 – Jun 30 (28,411)   test Jul 01 – Aug 31 (9,540)
fold_2:   train Jan 01 – Jul 31 (33,258)   test Aug 01 – Sep 30 (9,299)
fold_3:   train Jan 01 – Aug 31 (37,951)   test Sep 01 – Oct 31 (9,380)
```

Rolling origin. Each fold trains on everything up to its cut-off and tests on the
two months that follow, so all three rehearse the same two-month forward jump the
real task demands. `assert_no_leakage` runs on every split and raises if any test
date reaches back into training.

`random_split` exists in `src/validation/splitters.py` but is used only for the
comparison in the report, and logs a warning when called.

Once validation is done the final model refits on all 47,331 cleaned rows,
holding back September and October would waste the two months closest to what is
being predicted.

### Model

LightGBM on `log(rate per mile)`. The raw target is right-skewed (1.90) and close
to symmetric in logs (−0.49), so this stops long hauls dominating the error. A
smearing factor corrects the bias introduced by exponentiating back to dollars.

Feature importance on the final model:

```
great_circle    27.7%
distance        27.4%
equipment_code  19.5%
log_distance     5.5%
weight           4.4%
seasonal_index   3.7%
```

Distance and equipment carry roughly 80% of the signal. The seasonal component is
real but modest, which is why a fixed load's price moves only slightly across
December.

## Decisions tested rather than assumed

Three choices were made by running the alternatives and comparing, not by
argument. Each is recorded in `config/config.yaml` next to the setting it
controls.

**Seasonal curve as an offset, or as a feature?** Forcing the fitted curve through
as an offset guarantees it comes through at full strength. It also lost on every
fold — mean RMSE 108.02 against 88.91. The capability is still in the code
(`model.seasonal_offset`) but is off by default.

**Day-of-week features?** The weekly spread is only 2%. Removing them improved
mean RMSE from 91.87 to 88.91 and won on two of three folds, so they are off.

**How much seasonality to force?** There is a direct trade-off between the
accuracy score and how much the December curve moves:

| Configuration | Mean RMSE | Dec trend | Dec spread |
|---|---|---|---|
| Weekday + Fourier + curve | 91.87 | +$5.45 | $21.67 |
| **Fourier + curve** | **88.91** | +$0.82 | $9.77 |
| Curve only | 111.48 | −$36.87 | $36.97 |

The middle row is the default, because the accuracy score is what is graded. The
consequence is a December chart that varies by 1.23% rather than showing a
pronounced trend, which is an honest reflection of what the data supports.

## Project structure

```
freight-rate-prediction/
├── config/
│   ├── config.yaml              every setting, with the reasoning beside it
│   └── logging.yaml             console and file logging
├── data/
│   ├── train-test.csv           provided — labelled development data
│   ├── validation.csv           provided — 12,000 loads to price
│   ├── december_chart_inputs.csv  provided — filled in place by predict.py
│   └── 02-preprocessed/ 03-features/ 04-predictions/
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
├── tests/                       47 tests
├── models/                      trained bundle and metadata
├── reports/figures/             EDA, cleaning, and architecture figures
├── score.py                     provided — unmodified
├── Makefile
├── pytest.ini
└── requirements.txt
```

## Testing

```bash
pytest                       # 47 tests
pytest -m "not integration"  # 44 tests, no data files needed
```

Four areas, each guarding a specific failure:

- **Cleaning** : sign flips recovered, medians reused rather than refitted, and
  no row ever dropped at prediction time
- **Features** : unseen cities produce a code rather than a NaN, seasonal values
  stay defined and in range for December, no month column reaches the model
- **Splitters** : no fold leaks, and the guard rejects a random split
- **Submission** : mirrors every rule in `score.py`

The suite was mutation-tested: four deliberate bugs were introduced to check the
tests would catch them. The first attempt caught three of four, and the test that
missed one was rewritten until it did.

## Notes

- Predictions are guaranteed strictly positive, as `score.py` requires.
- `models/metadata.json` records what was trained, when, on which rows, and how it
  scored, so any prediction can be traced back to the run that produced it.
- `models/training_results.json` holds every score from the last run, so the
  report quotes the run rather than numbers copied by hand.
- Intermediate artifacts are gitignored; the model bundle, submission, and chart
  are committed so a reviewer can find them without running anything.