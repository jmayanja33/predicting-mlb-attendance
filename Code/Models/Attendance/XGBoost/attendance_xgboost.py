import pandas as pd
import numpy as np
import pickle
import yaml
import xgboost as xgb
import dataframe_image as dfi
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from root_path import ROOT_PATH


def calculate_adj_r2(r2, data):
    """
    Function to calculate adjusted r2
    :param r2: Previously calculated regular r2 number
    :param data: Dataframe from which r^2 was calculated
    :return:  AAdjusted r2 value
    """
    df = pd.DataFrame(data)
    num_observations = len(df)
    num_features = len(df.columns)
    return 1 - (1-r2) * (num_observations-1)/(num_observations-num_features-1)


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
full_data = full_data[full_data["Year"] != 2020]  # Remove 2020 data points from full dataset
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
plt.savefig(f"{ROOT_PATH}/Visualizations/AttendanceXGBFeatureImportance.png")

# Select features using threshold of 25
print("- Filtering data to only include selected features")
significant_features = SelectFromModel(feature_model, threshold=0.02, prefit=True)
significant_feature_names = [X_train.columns[i] for i in significant_features.get_support(indices=True)]

# Update X_train and X_test with selected features
X_train = significant_features.transform(X_train)
X_test = significant_features.transform(X_test)

# Create xgboost D-matrices
print("- Creating D-matrices and setting parameter evaluations for cross validation")
d_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
d_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Create dictionary of potential parameters for testing in cross validation
param_tuning = {
    "max_depth": np.arange(3, 10),
    "learning_rate": np.arange(0.1, 1, 0.1),
    "n_estimators": np.arange(100, 1000, 100),
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
# Make predictions
train_predictions = final_model.predict(X_train)
test_predictions = final_model.predict(X_test)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

# Calculate R-squared
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Calculate Adj. R-Squared
train_adj_r2 = calculate_adj_r2(train_r2, X_train)
test_adj_r2 = calculate_adj_r2(test_r2, X_test)

# Save Model
print("- Saving model and results")
final_model.save_model("XGBoost/AttendanceXGBoostModel.json")

# Write evaluation stats to a file
file = open('XGBoost/AttendanceXGBEval.txt', 'w')
file.write(f"""- Selected Features: {significant_feature_names}
- Best Params: {best_params}\n\n- Training RMSE: {train_rmse}\n- Training R-Squared: {train_r2}\n- Training Adj. R-squared: {train_adj_r2}
\n- Test RMSE: {test_rmse}\n- Test R-squared: {test_r2}\n- Test Adj R-Squared: {test_adj_r2}""")
file.close()

# Write feature names to a file
print("- Writing feature names and importance to a file")
with open("XGBoost/feature_names.pkl", "wb") as pklfile:
    pickle.dump(significant_feature_names, pklfile)
    pklfile.close()

# Create ordered dictionary of feature importance and write to file
importance_vals = feature_model.feature_importances_
importance_dict = dict(sorted({feature_model.feature_names_in_[i]: importance_vals[i] for i in range(len(importance_vals))}.items(),
                              key=lambda x: x[1], reverse=True))

importance_df = pd.DataFrame([(feature, importance_dict[feature]) for feature in importance_dict.keys()],
                             columns=["Feature", "Importance"])
importance_df = importance_df[importance_df["Importance"] >= 0.02]
importance_df.index += 1
dfi.export(importance_df, f"{ROOT_PATH}/Visualizations/SelectedFeatureImportance.png")
