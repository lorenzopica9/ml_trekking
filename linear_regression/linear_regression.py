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
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mpl_toolkits.mplot3d import Axes3D

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
    x_const = sm.add_constant(x_train)

    # alternative : normalize features values in 0-1 range
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(x)

    model = sm.OLS(y_train, x_const)
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
    x_grid = np.linspace(x_train.min(), x_train.max(), 200)
    pred = results.get_prediction(sm.add_constant(x_grid))
    sf = pred.summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_train, y_train, s=20, label='Data')
    ax.plot(x_grid, sf['mean'], color='C1', label='Fit')
    ax.fill_between(x_grid, sf['mean_ci_lower'], sf['mean_ci_upper'], color='C1', alpha=0.3, label='CI @ 95%')
    ax.set_xlabel(feature)

    ax.set_ylabel('time')
    ax.legend()
    fig.tight_layout()

    out_dir_fig = f'./output/figs/'
    os.makedirs(out_dir_fig, exist_ok=True)
    fig.savefig(out_dir_fig+'fit_'+feature+'.png', dpi=200)


def variables_selection(df, features, y_var, r_seed, fname, sel_type, test_size=0.2):
    x = df_trails[features].to_numpy()
    y = df_trails[y_var].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=r_seed)

    # Define model
    model = LinearRegression()

    # run selection
    sfs = SequentialFeatureSelector(model, n_features_to_select=2, direction=sel_type)
    sfs.fit(x_train, y_train)
    result = sfs.get_support()
    print_to_file(fname, 'Features '+sel_type+' selection')
    print_to_file(fname, 'Features: ' + ', '.join(features))
    print_to_file(fname, 'Results : ' + ', '.join(str(r) for r in result))
    


def multiple_linear_regression(df, features, y_var, r_seed, fname, test_size=0.2):
    #Define x and y for the regression
    x = df_trails[features].to_numpy()
    y = df_trails[y_var].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=r_seed)
    x_const = sm.add_constant(x_train)

    # define train df for later plotting
    df_train = pd.DataFrame(x_train, columns=features)
    df_train[y_var] = y_train 

    model = sm.OLS(y_train, x_const)
    results = model.fit()

    print_to_file(fname, '')
    print_to_file(fname, 'Multiple linear regression: y=' + y_var + ', x=' + ', '.join(features))

    # get confidence intervals
    conf_int = results.conf_int(alpha=0.05)

    # print results for intercept
    print_to_file(fname, 'Intercept = {:.2f} +- {:.2f}. t-statistic = {:.2f}, p-value = {:.3f}'.format(results.params[0], results.bse[0], results.tvalues[0], results.pvalues[0]))

    # print results for features
    for i in range(1, len(features)+1):
        print_to_file(fname, 'Param_'+features[i-1]+' = {:.2f} +- {:.2f}. t-statistic = {:.2f}, p-value = {:.3f}'.format(results.params[i], results.bse[i], results.tvalues[i], results.pvalues[i]))
        # confidence intervals
        print_to_file(fname, 'Param_'+features[i-1]+' CI @ 95% = [{:.2f}, {:.2f}]'.format(conf_int[i, 0], conf_int[i,1]))

    # model accuracy
    rse = np.sqrt(results.scale)
    r2 = results.rsquared
    f_stat = results.fvalue
    f_stat_pv = results.f_pvalue

    print_to_file(fname, 'RSE = {:.2f}'.format(rse))
    print_to_file(fname, 'R2 = {:.2f}. (0: linear fit explains data variation)'.format(r2))
    print_to_file(fname, 'F-stat = {:.2f}, p-value = {:.2f}'.format(f_stat, f_stat_pv))
    print_to_file(fname, '')

    if(len(features) == 2):
        n_grid=30

        # 2D grid on plane (x_train, y_train)
        # linspace of each feature, on interval observed on data
        x_range = np.linspace(df_train[features[0]].min(), df_train[features[0]].max(), n_grid)
        y_range = np.linspace(df_train[features[1]].min(), df_train[features[1]].max(), n_grid)

        # get 2D matrices from 1D vectors
        # xx[i,j], yy[i,j] are (i,j) 2D coordinates of grid
        xx, yy = np.meshgrid(x_range, y_range)

        # build df for prediction - 1D vector from 2D grid as get_prediction takes a vector
        X_pred = pd.DataFrame({
            features[0]: xx.ravel(),
            features[1]: yy.ravel(),
        })

        # has_constant='add' to add the intercept column
        pred = results.get_prediction(sm.add_constant(X_pred, has_constant='add'))

        # convert predicted_mean (1D) to 2D matrix in order to plot the fit
        zz = pred.predicted_mean.reshape(xx.shape)

        # 3D fig
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # observed points: uno scatter in 3D, un punto per riga del DataFrame
        ax.scatter(df_train[features[0]], df_train[features[1]], df_train[y_var], s=20, color='C0', label='Dati')

        # superficie di fit: il piano (o superficie) previsto dal modello
        ax.plot_surface(xx, yy, zz, color='C1', alpha=0.4)

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(y_var)
        fig.tight_layout()

        out_dir_fig = f'./output/figs/'
        os.makedirs(out_dir_fig, exist_ok=True)
        fname = f'{out_dir_fig}fit3d_{features[0]}_{features[1]}.png'
        fig.savefig(fname, dpi=200)
        plt.close(fig)




#out_fname = f'report_{datetime.now():%Y-%m-%d_%H-%M}'
out_dir = f'./output/'
out_fname = out_dir+f'report.md'

os.makedirs(out_dir, exist_ok=True)
with open(out_fname, 'w', encoding='utf-8') as f:
    f.write('Starting trails analysis.\n')

# file to open containing data
in_fname = '../create_dataset/gpx_files/trails.csv'
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

# perform linear regression on each single feature
linear_regression_1d(df_trails, 'length', 'time', r_seed=42, fname=out_fname, test_size=0.2)
linear_regression_1d(df_trails, 'uphill', 'time', r_seed=42, fname=out_fname, test_size=0.2)
linear_regression_1d(df_trails, 'downhill', 'time', r_seed=42, fname=out_fname, test_size=0.2)

# choose variables to include in multiple linear regression
variables_selection(df_trails, ['length', 'uphill', 'downhill'], 'time', r_seed=42, fname=out_fname, sel_type='forward', test_size=0.2)
variables_selection(df_trails, ['length', 'uphill', 'downhill'], 'time', r_seed=42, fname=out_fname, sel_type='backward', test_size=0.2)

# multiple linear regression
multiple_linear_regression(df_trails, ['length', 'uphill', 'downhill'], 'time', r_seed=42, fname=out_fname, test_size=0.2)
multiple_linear_regression(df_trails, ['length', 'downhill'], 'time', r_seed=42, fname=out_fname, test_size=0.2)
