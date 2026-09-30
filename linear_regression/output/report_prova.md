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
Intercept = 5597.00 +- 1470.24. t-statistic = 3.81, p-value = 0.000
Param_length = 0.71 +- 0.17. t-statistic = 4.26, p-value = 0.000
Param_length CI @ 95% = [0.38, 1.04]
Param_uphill = 3.91 +- 2.39. t-statistic = 1.63, p-value = 0.106
Param_uphill CI @ 95% = [-0.85, 8.67]
Param_downhill = 3.64 +- 2.38. t-statistic = 1.53, p-value = 0.130
Param_downhill CI @ 95% = [-1.10, 8.39]
RSE = 4897.42
R2 = 0.59. (0: linear fit explains data variation)
F-stat = 36.99, p-value = 0.00


Multiple linear regression: y=time, x=length, downhill
Intercept = 6117.19 +- 1450.84. t-statistic = 4.22, p-value = 0.000
Param_length = 0.74 +- 0.17. t-statistic = 4.41, p-value = 0.000
Param_length CI @ 95% = [0.41, 1.07]
Param_downhill = 6.37 +- 1.72. t-statistic = 3.70, p-value = 0.000
Param_downhill CI @ 95% = [2.94, 9.79]
RSE = 4950.25
R2 = 0.58. (0: linear fit explains data variation)
F-stat = 53.00, p-value = 0.00

