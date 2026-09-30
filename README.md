# ml_trekking

A small machine learning project studying how the characteristics of a hike
(distance, elevation gain/loss, whether it's a loop) relate to its duration,
using real GPX tracks from hikes in Italy.

The project is organized in stages:

1. **`create_dataset/`** — parses GPX files from real hikes and extracts, for
   each one: length, duration, elevation gain, elevation loss, and whether
   the route is a loop. The result is collected into a single CSV file.
2. **`linear_regression/`** — fits and evaluates regression models to predict
   hiking time from those features, using `scikit-learn` and `statsmodels`.

## Repository structure

```
ml_trekking/
├── create_dataset/
│   ├── gpx_files/          # raw GPX tracks used as input
│   ├── read_gpx.py         # parses the GPX files and builds the trails CSV
│   └── ...                 # output CSV with one row per hike
├── linear_regression/
│   ├── linear_regression.py  # simple/multiple regression, feature selection, plots
│   └── output/
│       ├── report_prova.md   # text report of the regression runs
│       └── figs/             # plots (fit lines, 3D surfaces, etc.)
├── .gitignore
└── README.md
```

## Workflow

```bash
cd create_dataset
python read_gpx.py         

cd ../linear_regression
python linear_regression.py 
```

## Project background

This project started as a personal exercise to apply classical regression
techniques (OLS, hypothesis testing, confidence intervals, feature selection,
diagnostics) to a real, non-trivial dataset built from one's own hiking
tracks, moving beyond point predictions towards a full statistical
understanding of the fit — uncertainties, significance of each feature, and
goodness of fit.

## License

No license is currently set for this repository.