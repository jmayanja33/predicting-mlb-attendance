import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from root_path import ROOT_PATH

# Load in training and test sets
print("- PROGRESS:\n")
print("- Loading Training/Test Sets")
full_data = pd.read_csv(f"{ROOT_PATH}/Data/CSVData/cleaned_current_markets_data.csv")
training_df = pd.read_csv(f"{ROOT_PATH}/Data/SplitData/TrainingSet.csv")
test_df = pd.read_csv(f"{ROOT_PATH}/Data/SplitData/TestSet.csv")

# Remove error features from data sets
print("- Removing unnecessary features from data (margin of error columns)")
train_cols = [col for col in training_df.columns if "Margin of Error" not in col]
test_cols = [col for col in test_df.columns if "Margin of Error" not in col]

# Remove teams with 2 markets
full_data = full_data[full_data["Year"] != 2020]
full_data = full_data[full_data["NumTeams"] == 1]
full_data.reset_index()
training_df = training_df[training_df["NumTeams"] == 1]
training_df.reset_index()
test_df = test_df[test_df["NumTeams"] == 1]
test_df.reset_index()

# Extract features and targets from each set
X_train, y_train = training_df[train_cols].drop(['Attendance', 'Valuation', 'Market', 'Year'], axis=1), training_df[['Attendance']]
X_test, y_test = test_df[test_cols].drop(['Attendance', 'Valuation', 'Market', 'Year'], axis=1), test_df[['Attendance']]

# Create model with no training data to find feature importance
print("- Training XGB model on whole data set to find feature importance")
feature_model = XGBRegressor()
feature_model.fit(full_data[train_cols].drop(['Attendance', 'Valuation', 'Market', 'Year'], axis=1), full_data[["Attendance"]])

# Find and plot feature importance
print("- Plotting feature importance")
feature_importance = sorted(feature_model.feature_importances_, reverse=True)
plt.scatter(range(0, len(feature_importance)), feature_importance)
plt.xlabel("Feature Number")
plt.ylabel("Importance")
plt.title("Attendance XGB Model Feature Importance")
# plt.show()
plt.savefig(f"{ROOT_PATH}/Visualizations/AttendanceXGBFeatureImportance.png")

# Plot zoomed feature importance with names
# Create ordered dictionary of feature importance
importance_vals = feature_model.feature_importances_
importance_dict = dict(sorted({feature_model.feature_names_in_[i]: importance_vals[i] for i in range(len(importance_vals))}.items(),
                              key=lambda x: x[1], reverse=True))
plt.scatter(range(0, 25), feature_importance[0:25])
x_ticks = list(importance_dict.keys())[0:25]
plt.xticks(range(0, 25), x_ticks, rotation=270)
plt.xlabel("Feature")
plt.ylabel("Importance")
# plt.show()
plt.savefig(f"{ROOT_PATH}/Visualizations/AttendanceXGBFeatureImportanceTop25.png")

# Select features using threshold of 25
print("- Filtering data to only include selected features")
significant_features = SelectFromModel(feature_model, threshold=0.02, prefit=True)
significant_feature_names = [X_train.columns[i] for i in significant_features.get_support(indices=True)]
# significant_feature_cols = {str(i): X_train.columns[i] for i in significant_features.get_support(indices=True)}

# Update X_train and X_test with selected features
X_train = significant_features.transform(X_train)
X_test = significant_features.transform(X_test)

# Create xgboost D-matrices
print("- Creating D-matrices and setting parameters evaluations for cross validation")
d_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
d_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Create dictionary of potential parameters for testing in cross validation
param_tuning = {
    "max_depth": np.arange(3, 10),
    "learning_rate": np.arange(0.1, 1, 0.1),
    "n_estimators": np.arange(100, 1000, 100),
    # "colsample_bytree": np.arange(0.1, 1.1, 0.1),
    "gamma": np.arange(0, 5)
}

# Use grid search to perform k-fold cross validation with k=5 to find best parameters
print("- Performing 5 fold cross validation:")
xgb_object = XGBRegressor(seed=33)
params_model = GridSearchCV(estimator=xgb_object, param_grid=param_tuning, scoring="neg_mean_squared_error", verbose=10)
params_model.fit(X_train, y_train)
best_params = params_model.best_params_

# Create final model with best parameters
print("- Creating final model with best parameters from cross validation")
final_model = XGBRegressor(**best_params, random_state=33)
final_model.fit(X_train, y_train)

# Evaluate model
print("- Evaluating model performance")
train_predictions = final_model.predict(X_train)
test_predictions = final_model.predict(X_test)
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

# Save Model
print("- Saving model and results")
final_model.save_model("AttendanceXGBoostModel.json")

# Write file to save stats
file = open('AttendanceXGBEval.txt', 'w')
file.write(f"- Selected Features: {significant_feature_names}\n- Best Params: {best_params}\n- Training RMSE: {train_rmse}\n- Test RMSE: {test_rmse}")
file.close()
