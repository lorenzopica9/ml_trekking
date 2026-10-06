Starting trails analysis.
Opening ../create_dataset/gpx_files/trails.csv file.

Linear regression: y=time, x=length
Intercept = 6061.52 +- 1733.21. t-statistic = 3.50, p-value = 0.001
Slope(length) = 1.15 +- 0.15. t-statistic = 7.66, p-value = 0.000
Intercept CI @ 95% = [2596.90, 9526.15]
Slope(length) CI @ 95% = [0.85, 1.45]
RSE = 5230.67
R2 = 0.49. (0: linear fit explains data variation)


Linear regression: y=time, x=uphill
Intercept = 8408.02 +- 1592.02. t-statistic = 5.28, p-value = 0.000
Slope(uphill) = 14.44 +- 2.09. t-statistic = 6.92, p-value = 0.000
Intercept CI @ 95% = [5225.63, 11590.42]
Slope(uphill) CI @ 95% = [10.27, 18.61]
RSE = 5480.39
R2 = 0.44. (0: linear fit explains data variation)


Linear regression: y=time, x=downhill
Intercept = 9006.78 +- 1457.66. t-statistic = 6.18, p-value = 0.000
Slope(downhill) = 14.07 +- 1.95. t-statistic = 7.23, p-value = 0.000
Intercept CI @ 95% = [6092.95, 11920.61]
Slope(downhill) CI @ 95% = [10.18, 17.96]
RSE = 5375.71
R2 = 0.46. (0: linear fit explains data variation)


Linear regression: y=time, x=circular_1
Intercept = 14571.67 +- 4184.49. t-statistic = 3.48, p-value = 0.001
Slope(circular_1) = 3972.69 +- 4286.15. t-statistic = 0.93, p-value = 0.358
Intercept CI @ 95% = [6206.99, 22936.34]
Slope(circular_1) CI @ 95% = [-4595.20, 12540.59]
RSE = 7247.75
R2 = 0.01. (0: linear fit explains data variation)

Features forward selection
Features: length, uphill, downhill, circular_1
Results : True, False, True, False
Features backward selection
Features: length, uphill, downhill, circular_1
Results : True, False, True, False

Multiple linear regression: y=time, x=length, downhill
Intercept = 5229.66 +- 1619.45. t-statistic = 3.23, p-value = 0.002
Param_length = 0.73 +- 0.18. t-statistic = 3.97, p-value = 0.000
Param_length CI @ 95% = [0.36, 1.10]
Param_downhill = 7.96 +- 2.33. t-statistic = 3.41, p-value = 0.001
Param_downhill CI @ 95% = [3.30, 12.62]
RSE = 4831.76
R2 = 0.57. (0: linear fit explains data variation)
F-stat = 40.22, p-value = 0.00

