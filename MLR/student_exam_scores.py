import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
dataframe = pd.read_csv('./datasets/student_data_scores.csv', encoding='latin1')

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

# Train the MLR model
X = dataframe.drop(columns=["MathScore", "ReadingScore", "WritingScore"])
# Target
y = dataframe[["MathScore", "ReadingScore", "WritingScore"]]

# Convert categorical features
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Init and train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Feature importantce
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)