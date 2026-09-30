# linear_regression

Linear regression analysis of the trekking trail dataset (built by
`create_dataset/read_gpx.py`), predicting hiking `time` from trail features.

The script covers **simple linear regression** (one feature at a time),
**feature selection**, and **multiple linear regression** (several features
at once), using `statsmodels.OLS` for full statistical inference rather than
just point predictions.

## What it does

Running `linear_regression.py`:

1. Loads the trails CSV built by `create_dataset/read_gpx.py`, prints basic
   descriptive statistics (mean, std, min, max) for `length`, `time`,
   `uphill`, and `downhill`, and drops rows with missing values.
2. **Simple regression** (`linear_regression_1d`) — fits `time` against each
   of `length`, `uphill`, and `downhill` individually, and for each one:
   - reports the intercept and slope with their standard errors,
     t-statistics, and p-values
   - reports 95% confidence intervals for both coefficients
   - reports RSE and R²
   - saves a plot with the data, the fitted line, and its 95% confidence
     band to `output/figs/fit_<feature>.png`
3. **Feature selection** (`variables_selection`) — runs
   `sklearn.SequentialFeatureSelector` in both `forward` and `backward` mode
   over `length`, `uphill`, `downhill`, to see which pair of features it
   picks as most predictive.
4. **Multiple regression** (`multiple_linear_regression`) — fits `time`
   against several features at once (all three, and then just `length` and
   `downhill`), reporting per-feature coefficients, standard errors,
   t-statistics, p-values, and confidence intervals, plus overall RSE, R²,
   and F-statistic. When exactly two features are used, it also saves a 3D
   plot of the data and the fitted regression plane to
   `output/figs/fit3d_<feature1>_<feature2>.png`.

All the text results are appended to a single report file (`output/report_prova.md`
in the current version — see note below).

## Usage

```bash
cd linear_regression
python linear_regression.py
```

This expects the trails CSV to be available at the relative path used in the
script (currently `../create_dataset/gpx_apuane/trails.csv` — adjust this if
the CSV produced by `read_gpx.py` lives elsewhere).

## Output

- `output/report.md` — text report with coefficients, uncertainties,
  t-statistics, p-values, confidence intervals, RSE, R², and F-statistic for
  every regression run
- `output/figs/` — one PNG per simple regression (`fit_<feature>.png`) and
  per two-feature multiple regression (`fit3d_<feature1>_<feature2>.png`)