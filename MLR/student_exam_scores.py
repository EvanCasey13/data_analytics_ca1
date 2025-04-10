import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Init and train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#test
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Mean Squared Error: {test_mse}")
print(f"R-squared: {test_r2}")
print(f"Mean Absolute Error: {test_mae}")

#train
print()
print("Train metrics")
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)

print(f"Mean Squared Error: {train_mse}")
print(f"R-squared: {train_r2}")
print(f"Mean Absolute Error: {train_mae}")

# Feature importantce
coeff_df = pd.DataFrame(model.coef_, columns=X.columns, index=["MathScore", "ReadingScore", "WritingScore"])
print("\nFeature Importance:")
print(coeff_df.T) # features as rows

# Cross validation for model performance
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation: {abs(cv_scores.mean())}")

# Outliers boxplot
sns.boxplot(data=dataframe[['MathScore', 'ReadingScore', 'WritingScore']])
plt.title('Boxplot of Scores')
plt.show()
