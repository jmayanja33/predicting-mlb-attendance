import pandas as pd
import pickle
from xgboost import XGBRegressor
from root_path import ROOT_PATH


def rename_columns(df, reverse=False):
    """Function to rename columns to match model data frame"""
    columns = df.columns
    column_names = dict()
    for i in range(len(columns)):
        column = columns[i]
        column_names[column] = i
    return df.rename(columns=column_names)


# Load data
print("PROGRESS:\n")
print("- Loading Data")
new_data = pd.read_csv(f"{ROOT_PATH}/Data/CSVData/cleaned_new_markets_data.csv")
with open("feature_names.pkl", "rb") as pklfile:
    significant_features = pickle.load(pklfile)
    pklfile.close()

# Format data
print("- Formatting data")
formatted_new_data = new_data[significant_features]
formatted_new_data = rename_columns(formatted_new_data)

# Load model
print("- Loading model")
attendance_model = XGBRegressor()
attendance_model.load_model("AttendanceXGBoostModel.json")

# Make predictions
print("- Making predictions")
predictions = attendance_model.predict(formatted_new_data)

# Save predictions to csv file
print("- Saving predictions")
new_data["Predicted Attendance"] = predictions
new_data.to_csv(f"{ROOT_PATH}/Data/Predictions/PredictedNewMarketAttendanceXGBoost.csv")
