# create_dataset

Builds the dataset used by the regression analysis, starting from real GPX
tracks of hikes.

## What it does

`read_gpx.py` reads the GPX files in `gpx_files/` and, for each hike,
extracts:

- **length** — total distance of the track
- **time** — duration of the hike
- **uphill** — cumulative elevation gain
- **downhill** — cumulative elevation loss
- **is a loop** — whether the route starts and ends at (approximately) the
  same point

These are collected into a single CSV file, one row per hike, which is the
input consumed by `linear_regression/linear_regression.py`.

## Usage

```
python read_gpx.py
```

## Input data

`gpx_files/` contains the raw `.gpx` tracks recorded from each hike. Add new
hikes by dropping their `.gpx` file in this folder and re-running
`read_gpx.py`.