# Freight Rate Prediction

Predicts the posted rate of truckload freight loads. The final model is a hybrid: a robust linear model on log rate
(with spline curves and a time trend frozen at the last training day), plus LightGBM trained on its residuals.
Rolling monthly validation gives a mean absolute error of about $49 one month ahead and $56 two months ahead,
against $69 and $68 for plain LightGBM.

## Setup

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
# Train on all labeled data, write validation_predictions.csv and fill data/december_chart_inputs.csv
python predict.py

# Validate both output files and create scorer_results/candidate_december.png
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Optional:

```bash
# Rolling monthly validation (results in reports/cv_folds.csv and reports/cv_summary.csv)
python validate.py                      # main model comparison
python validate.py best_variants        # one-change checks on the final model
python validate.py --gap 1 <names...>   # two-months-ahead folds (reports/*_gap1.csv)
python validate.py all                  # every registered experiment

# Exploratory data analysis on the raw files
python -m jupyter notebook notebooks/01_eda.ipynb
```

CatBoost experiments take about 6 minutes; everything else runs in under 1.5 minutes per experiment.

## Project layout

| Path | Contents |
|---|---|
| `data/` | Provided input files |
| `notebooks/01_eda.ipynb` | Exploration of the raw data: missing values, correlations, outliers, time patterns, findings |
| `src/clean.py` | Fixed cleaning rules, weight imputer, city lookup, December input preparation, data-quality report |
| `src/features.py` | Row features (haversine distance, distance ratio, trend, day of week) and out-of-fold lane/city encodings |
| `src/model.py` | Median baseline, configurable model (`lgbm`, `catboost`, `linear`, `hybrid`, ...), blends, `FINAL_CONFIG` |
| `src/validation.py` | Expanding-window monthly folds and dollar metrics |
| `validate.py` | Experiment registry and runner |
| `predict.py` | Final training and submission files |
| `score.py` | Provided scorer |

## Approach in brief

- **Cleaning** (never drops rows): absolute value of negative weights; flag weights capped at 47,500 lb; fill missing
  weight with the equipment median from the training months; fill missing `market_index` with the same-date median.
- **Validation**: expanding-window folds, training on every month before each holdout month (July to October), plus
  a version with a one-month gap matching the two-month horizon of the final task. All fitted steps are fitted inside
  each fold. Errors are reported in dollars on all rows and on rows with plausible rates, because about 650 suspected
  corrupt rates dominate RMSE.
- **December chart**: coordinates come from the city lookup, and `market_index` / `quote_signal` are the same-date
  medians from `validation.csv`, which covers every December day.
