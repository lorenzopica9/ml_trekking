import math
from datetime import datetime
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels .api as sm

from statsmodels .stats. outliers_influence \
import variance_inflation_factor as VIF
from statsmodels .stats.anova import anova_lm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler # scale data as preprocessing
from sklearn.preprocessing import MinMaxScaler   # scale data in 0-1 range

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# define small function for clean info display
def printinfo(text):
    if not isinstance(text, str):
        raise TypeError('printinfo only accepts strings as arguments.')
    if text == '':
        print('#' * 100 + '\n')
    else:
        print('#' * math.floor((100 - len(text) - 2)/2) + ' ' + text + ' ' + '#' * math.ceil((100 - len(text) - 2)/2))


def print_to_file(file, text):
    with open(out_fname, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

def linear_regression_1d(df, feature, y_var, r_seed, fname, test_size=0.2):
    #Define x and y for the regression
    x = df_trails[[feature]].to_numpy() # to numpy to have a 1D array, otherwise is a 1D df
    # x = df_trails[['length', 'uphill', 'downhill']] # Multiple features
    y = df_trails[y_var].to_numpy()

    # split train-test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=r_seed)

    # start regression

    # normalize features values - scale accordingly
    #scaler = StandardScaler()
    #x_scaled = scaler.fit_transform(x)
    # add intercept column for statsmodel
    x_const = sm.add_constant(x)

    # alternative : normalize features values in 0-1 range
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(x)

    model = sm.OLS(y, x_const)
    results = model.fit()
    #print(results.summary())

    print_to_file(fname, '')
    print_to_file(fname, 'Linear regression: y=' + y_var + ', x=' + feature)

    # compute and print values, uncertainties, t-statistics and p-values
    print_to_file(fname, 'Intercept = {:.2f} +- {:.2f}. t-statistic = {:.2f}, p-value = {:.3f}'.format(results.params[0], results.bse[0], results.tvalues[0], results.pvalues[0]))
    print_to_file(fname, 'Slope('+feature+') = {:.2f} +- {:.2f}. t-statistic = {:.2f}, p-value = {:.3f}'.format(results.params[1], results.bse[1], results.tvalues[1], results.pvalues[1]))
    
    # confidence intervals
    conf_int = results.conf_int(alpha=0.05)
    print_to_file(fname, 'Intercept CI @ 95% = [{:.2f}, {:.2f}]'.format(conf_int[0, 0], conf_int[0, 1]))
    print_to_file(fname, 'Slope('+feature+') CI @ 95% = [{:.2f}, {:.2f}]'.format(conf_int[1, 0], conf_int[1,1]))

    # model accuracy
    rse = np.sqrt(results.scale)
    r2 = results.rsquared

    print_to_file(fname, 'RSE = {:.2f}'.format(rse))
    print_to_file(fname, 'R2 = {:.2f}. (0: linear fit explains data variation)'.format(r2))
    print_to_file(fname, '')

    # plot results
    x_grid = np.linspace(x.min(), x.max(), 200)
    pred = results.get_prediction(sm.add_constant(x_grid))
    sf = pred.summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=20, label='Dati')
    ax.plot(x_grid, sf['mean'], color='C1', label='Fit OLS')
    ax.fill_between(x_grid, sf['mean_ci_lower'], sf['mean_ci_upper'], color='C1', alpha=0.3, label='IC 95% della retta')
    ax.set_xlabel('x')

    ax.set_ylabel('y')
    ax.legend()
    fig.tight_layout()

    out_dir_fig = f'./output/figs/'
    os.makedirs(out_dir_fig, exist_ok=True)
    fig.savefig(out_dir_fig+'fit_'+feature+'.png', dpi=200)



#out_fname = f'report_{datetime.now():%Y-%m-%d_%H-%M}'
out_dir = f'./output/'
out_fname = out_dir+f'report_prova.md'

os.makedirs(out_dir, exist_ok=True)
with open(out_fname, 'w', encoding='utf-8') as f:
    f.write('Starting trails analysis.\n')

# file to open containing data
in_fname = '../create_dataset/gpx_apuane/trails.csv'
df_trails = pd.read_csv(in_fname)

print_to_file(out_fname, 'Opening ' + in_fname + ' file.')

# print(df_trails)

#inspect dataset - to be completed
print('Info on trails length:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m'\
      .format(df_trails['length'].mean(), df_trails['length'].std(),\
              df_trails['length'].min(), df_trails['length'].max()))

print('Info on trails time:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m'\
      .format(df_trails['time'].mean(), df_trails['time'].std(),\
              df_trails['time'].min(), df_trails['time'].max()))

print('Info on trails uphill:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m'\
      .format(df_trails['uphill'].mean(), df_trails['uphill'].std(),\
              df_trails['uphill'].min(), df_trails['uphill'].max()))

print('Info on trails downhill:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m'\
      .format(df_trails['downhill'].mean(), df_trails['downhill'].std(),\
              df_trails['downhill'].min(), df_trails['downhill'].max()))


# one-hot encoding of categorical features
# df_trails = pd.get_dummies(df_trails, columns=['Education_Level'], drop_first=True)

#Cleanup dataset removing missing values
df_trails.dropna(inplace=True)

# fast check before proceeding with the regression
printinfo('Checking dataframe after cleanup, before regression.')
print(df_trails.info())  # Check column types and missing values
print(df_trails.describe())  # Get summary statistics
printinfo('')

linear_regression_1d(df_trails, 'length', 'time', r_seed=42, fname=out_fname, test_size=0.2)
linear_regression_1d(df_trails, 'uphill', 'time', r_seed=31, fname=out_fname, test_size=0.2)
linear_regression_1d(df_trails, 'downhill', 'time', r_seed=94, fname=out_fname, test_size=0.2)



# # make predictions on test data
# y_pred = model.predict(x_test)

# predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(predictions.head())

# # evaluate model prediction

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print('Mean Absolute Error (MAE):', mae)
# print('Mean Squared Error (MSE):', mse)
# print('R-squared Score:', r2)
