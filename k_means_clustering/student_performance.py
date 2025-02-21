from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Load the dataset
dataframe = pd.read_csv('./datasets/Student_performance.csv', encoding='latin1')
dataframe = dataframe.drop(columns=['roll_no'])
dataframe = pd.get_dummies(dataframe, columns=['race_ethnicity', 'parental_level_of_education', 'grade'], drop_first=True)

# Encoding categorical columns
categorical_columns = ['gender', 'test_preparation_course', 'math_score']
le = LabelEncoder()
for col in categorical_columns:
   dataframe[col] = le.fit_transform(dataframe[col])

# Drop missing values
dataframe.dropna(inplace=True)

X = dataframe.drop(columns=["total_score"])
y = dataframe["total_score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# k-NN model
knn = KNeighborsRegressor(n_neighbors=8)  
knn.fit(X_train_scaled, y_train)

# evaluate the model
y_pred = knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
