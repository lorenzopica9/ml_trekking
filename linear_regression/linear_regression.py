import math

import numpy as np
import pandas as pd
from matplotlib .pyplot import subplots

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
        raise TypeError("printinfo only accepts strings as arguments.")
    if text == '':
        print('#' * 100)
        print('#' * 100 + '\n')
    else:
        print('#' * 100)
        print('#' * math.floor((100 - len(text) - 2)/2) + ' ' + text + ' ' + '#' * math.ceil((100 - len(text) - 2)/2))
        print('#' * 100)

df_trails = pd.read_csv("../create_dataset/gpx_apuane/trails.csv")

# print(df_trails)

#inspect dataset
print("Info on trails length:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m"\
      .format(df_trails["length"].mean(), df_trails["length"].std(),\
              df_trails["length"].min(), df_trails["length"].max()))

print("Info on trails time:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m"\
      .format(df_trails["time"].mean(), df_trails["time"].std(),\
              df_trails["time"].min(), df_trails["time"].max()))

print("Info on trails uphill:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m"\
      .format(df_trails["uphill"].mean(), df_trails["uphill"].std(),\
              df_trails["uphill"].min(), df_trails["uphill"].max()))

print("Info on trails downhill:\n mean = {:.2f} m \n std = {:.2f} m \n min = {:.2f} m \n max = {:.2f} m"\
      .format(df_trails["downhill"].mean(), df_trails["downhill"].std(),\
              df_trails["downhill"].min(), df_trails["downhill"].max()))

#Cleanup dataset removing missing values
df_trails.dropna(inplace=True)

#Define x and y for the regression
#x = df_trails[['length']] #Just 1D
x = df_trails[['length', 'uphill', 'downhill']] # Multiple features
y = df_trails['time']

# one-hot encoding of categorical features
# df_trails = pd.get_dummies(df_trails, columns=['Education_Level'], drop_first=True)

# split train-test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# fast check before proceeding with the regression
printinfo('Checking dataframe after cleanup, before regression.')
print(df_trails.info())  # Check column types and missing values
print(df_trails.describe())  # Get summary statistics
printinfo('')

####################################################################################################
######################################### Start regression #########################################
####################################################################################################

# normalize features values - scale accordingly
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# alternative : normalize features values in 0-1 range
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# train the model
model = LinearRegression()
model.fit(x, y)

printinfo('Linear model training results')
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
printinfo('')

# make predictions on test data
y_pred = model.predict(x_test)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

# evaluate model prediction

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)
