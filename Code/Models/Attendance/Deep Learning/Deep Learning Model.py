import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Import training dataset
train_url = '/Users/anthony_palmeri/PycharmProjects/Team-116/Data/SplitData/TrainingSet.csv'
train_data = pd.read_csv(train_url)
# print(train_data.head())

# Import test dataset
test_url = '/Users/anthony_palmeri/PycharmProjects/Team-116/Data/SplitData/TestSet.csv'
test_data = pd.read_csv(test_url)
# print(test_data.head())

# Training data preprocessing
train_data = train_data[train_data['Year'] != 2020.0]
drop_columns = [col for col in train_data.columns if 'Error' in col or 'Past' in col]
drop_columns.extend(['Market', 'Year', 'Dome', 'NumTeams', 'Valuation'])
train_data = train_data.drop(columns=drop_columns)
train_x = train_data.drop('Attendance', axis=1)
train_y = train_data['Attendance']

# Testing data preprocessing
test_data = test_data[test_data['Year'] != 2020.0]
test_data = test_data.drop(columns=drop_columns)
test_x = test_data.drop('Attendance', axis=1)
test_y = test_data['Attendance']

# Scale data
# scaler = MinMaxScaler()
# train_x = scaler.fit_transform(train_x)
# test_x = scaler.fit_transform(test_x)

# Build model
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=50)

# Output
print(model)

# predict the values from your model
predictions = model.predict(test_x)
predictions = pd.Series(predictions.reshape(-1))

# print(predictions.shape)
# print(test_y.head())

# calculate residuals
# residuals = test_y - predictions
#
# # # plot residuals
# plt.figure(figsize=(10,6))
# plt.scatter(test_y, residuals)
# plt.title('Residual Plot')
# plt.xlabel('Observed')
# plt.ylabel('Residuals')
# plt.show()


# Get metrics
rmse = np.sqrt(mean_squared_error(test_y, predictions))
print("Root Mean Squared Error: ", rmse)

r2 = r2_score(test_y, predictions)
print("R-squared: ", r2)

n = test_y.shape[0]
p = test_x.shape[1]
adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
print("Adjusted R-squared: ", adj_r2)


# Bring in new markets
new_markets_url = '/Users/anthony_palmeri/PycharmProjects/Team-116/Data/CSVData/cleaned_new_markets_data.csv'
new_markets_data = pd.read_csv(new_markets_url)
print(new_markets_data.head())

# New market data preprocessing
new_markets_data_clean = new_markets_data[new_markets_data['Year'] != 2020.0]
drop_columns = [col for col in new_markets_data_clean.columns if 'Error' in col or 'Past' in col]
drop_columns.extend(['Market', 'Year', 'Dome', 'NumTeams'])
new_markets_data_clean = new_markets_data_clean.drop(columns=drop_columns)

# Predict
new_predictions = model.predict(new_markets_data_clean)
print(new_predictions)