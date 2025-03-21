from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
dataframe = pd.read_csv('./datasets/Student_performance.csv', encoding='latin1')
dataframe = dataframe.drop(columns=['roll_no'])
dataframe = pd.get_dummies(dataframe, columns=['race_ethnicity', 'parental_level_of_education', 'grade', 'gender', 'test_preparation_course', 'math_score'], drop_first=True)

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

# Use grid search wityh cross validate to determine best k value
knn = KNeighborsRegressor()  
param_grid = {'n_neighbors': range(1, 21)}

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

optimal_k = grid_search.best_params_['n_neighbors']
print(f"The optimal K value is: {optimal_k}\n")

# k-NN regression model
knn = KNeighborsRegressor(n_neighbors=optimal_k)  
knn.fit(X_train_scaled, y_train)

# evaluate the model
y_pred_test = knn.predict(X_test_scaled)
y_pred_train = knn.predict(X_train_scaled)

# test
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
evs_test = explained_variance_score(y_test, y_pred_test)

# train
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
evs_train = explained_variance_score(y_train, y_pred_train)

print("Test regression results")
print(f"Mean Squared error: {mse_test}")
print(f"R-squared: {r2_test}")
print(f"Mean absolute error: {mae_test}")
print(f"Root Mean squared error: {rmse_test}")
print(f"Explained variance score: {evs_test}\n")

print("Train regression results")
print(f"Mean Squared error: {mse_train}")
print(f"R-squared: {r2_train}")
print(f"Mean absolute error: {mae_train}")
print(f"Root Mean sqaured error: {rmse_train}")
print(f"Explained variance score: {evs_train}\n")