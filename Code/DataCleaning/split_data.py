import pandas as pd
from tqdm import tqdm
from root_path import ROOT_PATH


def update_test_year(year):
    """Function to track year for rotation"""
    if year == 2021:
        year = 2011
    elif year == 2019:
        year += 2
    else:
        year += 1
    return year


# Read in current market data and remove the year 2020
data = pd.read_csv(f"{ROOT_PATH}/Data/CSVData/cleaned_current_markets_data.csv")
data = data[data["Year"] != 2020]
data.reset_index(inplace=True, drop=True)

# Initialize list of markets, lists for train/test data points
markets = data["Market"].unique()
train_data = []
test_data = []

# Define the first two years to assign test data for the rotation
first_test_year = 2011
second_test_year = 2016

# Split data into train and test set using a rotation (80% train, 20% test)
print("PROGRESS:\n")
print("- Splitting data:")
for market in tqdm(markets):
    # Find all data points for the market
    market_df = data[data["Market"] == market]
    # Assign 8 data points to train set and 2 to test set
    for i in range(10):
        data_point = market_df.iloc[i]
        if data_point["Year"] in {first_test_year, second_test_year}:
            test_data.append(data_point)
        else:
            train_data.append(data_point)
    # Update years for test set
    first_test_year = update_test_year(first_test_year)
    second_test_year = update_test_year(second_test_year)

# Create data frames for each set and save to csv files
print(f"- Saving data to csv files: '{ROOT_PATH}/Data/SplitData/TrainingSet.csv' and '{ROOT_PATH}/Data/SplitData/TestSet.csv'")
training_set = pd.DataFrame(data=train_data)
test_set = pd.DataFrame(data=test_data)
training_set.to_csv(f"{ROOT_PATH}/Data/SplitData/TrainingSet.csv", index=False)
test_set.to_csv(f"{ROOT_PATH}/Data/SplitData/TestSet.csv", index=False)
