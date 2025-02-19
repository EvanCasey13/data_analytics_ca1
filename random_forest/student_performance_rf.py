import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

dataframe = pd.read_csv('./datasets/Student_performance.csv', encoding='latin1')

# Section 1 - Data Understanding section of CRISP-DM Process

# Describing the data
dataframe_info = dataframe.info()
print(dataframe_info)

# Data exploration
print(dataframe.head())
print(dataframe.tail()) 

print(dataframe.shape)

print(dataframe.columns)

print(dataframe.dtypes)

# Numerical statistics
#print(dataframe.describe())
print(dataframe.describe(include="all"))

# Null count
print(dataframe.isnull().sum())

# Verifying data quality
# Drop rows with a NA value
dataframe.dropna(inplace=True)
print(dataframe.info())

# Section 2 - Data cleaning and preparation 

# Feature selection
selected_features = ['gender', 'parental_level_of_education', 'lunch', 'test_preparation_course']
total_score = dataframe['total_score']

# Train the random forest model
label_encoder = LabelEncoder()
x_categorical = dataframe[selected_features].apply(label_encoder.fit_transform)
x_numerical = dataframe.select_dtypes(exclude=['object']).values
x = x_categorical
y = total_score.values  #Target variable

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

regressor.fit(x, y)

# OOB Score for model performance evaluation
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x)

mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')