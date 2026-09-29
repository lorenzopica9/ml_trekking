Starting trails analysis.
Opening ../create_dataset/gpx_apuane/trails.csv file.

Linear regression: y=time, x=length
Intercept = 6061.26 +- 1733.27. t-statistic = 3.50, p-value = 0.001
Slope(length) = 1.15 +- 0.15. t-statistic = 7.66, p-value = 0.000
Intercept CI @ 95% = [2596.50, 9526.02]
Slope(length) CI @ 95% = [0.85, 1.45]
RSE = 5230.72
R2 = 0.49. (0: linear fit explains data variation)


Linear regression: y=time, x=uphill
Intercept = 8402.82 +- 1593.33. t-statistic = 5.27, p-value = 0.000
Slope(uphill) = 14.44 +- 2.09. t-statistic = 6.92, p-value = 0.000
Intercept CI @ 95% = [5217.80, 11587.83]
Slope(uphill) CI @ 95% = [10.27, 18.61]
RSE = 5481.44
R2 = 0.44. (0: linear fit explains data variation)


Linear regression: y=time, x=downhill
Intercept = 9000.61 +- 1458.51. t-statistic = 6.17, p-value = 0.000
Slope(downhill) = 14.07 +- 1.95. t-statistic = 7.23, p-value = 0.000
Intercept CI @ 95% = [6085.10, 11916.12]
Slope(downhill) CI @ 95% = [10.18, 17.96]
RSE = 5375.86
R2 = 0.46. (0: linear fit explains data variation)

Features forward selection
Features: length, uphill, downhill
Results : True, False, True
Features backward selection
Features: length, uphill, downhill
Results : True, False, True

Multiple linear regression: y=time, x=length, uphill, downhill
