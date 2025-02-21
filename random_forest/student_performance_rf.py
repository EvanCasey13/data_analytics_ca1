import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
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

# Train the random forest model
X = dataframe.drop(columns=["total_score"])
y = dataframe["total_score"]

# Convert categorical features
categorical_columns = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])
    
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

regressor.fit(X_train, y_train)

# OOB Score for model performance evaluation
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

# Cross val score
scores = cross_val_score(regressor, X_test, y_test, cv=5, scoring='r2')
print(f'Cross validation r2 scores: {scores}')
print(f'Mean r2: {np.mean(scores)}')

# Feature importance
importances = list(regressor.feature_importances_)

# Display results
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df)

# Visualisations
# Feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
#plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
