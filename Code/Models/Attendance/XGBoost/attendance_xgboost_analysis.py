import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
from root_path import ROOT_PATH

# Load predictions
print("PROGRESS:\n")
print("- Loading current market data and predictions")
predictions = pd.read_csv(f"{ROOT_PATH}/Data/Predictions/PredictedNewMarketAttendanceXGBoost.csv")
predictions = predictions[predictions["Year"] != 2020]
predictions.reset_index(inplace=True, drop=True)

# Load current data
current_data = pd.read_csv(f"{ROOT_PATH}/Data/CSVData/cleaned_current_markets_data.csv")
current_data = current_data[current_data["Year"] != 2020]
current_data = current_data[current_data["NumTeams"] == 1]
current_data.reset_index(inplace=True, drop=True)

# Find per game margin of error
print("- Finding per game RMSE")
avg_attendance = round(current_data["Attendance"].mean()/81, 0)
rmse_avg_attendance = 268577.7372373305/81
error_file = open(f'{ROOT_PATH}/Code/Models/Attendance/XGBoost/AttendanceXGBError.txt', 'w')
error_file.write(f"""- Average MLB Per Game Attendance: {avg_attendance}\n- RMSE per Game: {rmse_avg_attendance}""")
error_file.close()

# Filter data
print("- Formatting data")
new_attendance_df = predictions[["Year", "Market", "Dome", "Predicted Attendance"]]
current_attendance_df = current_data[["Year", "Market", "Dome", "Attendance"]]

# Combine new and current attendance df
new_attendance_df_copy = new_attendance_df.rename(columns={"Predicted Attendance": "Attendance"})
full_attendance_df = pd.concat([current_attendance_df, new_attendance_df_copy], axis=0)
full_attendance_df = full_attendance_df.sort_values(by="Attendance", ascending=False)
full_attendance_df.reset_index(inplace=True, drop=True)
full_attendance_df.index += 1

# ANALYTICS FOR ONLY NEW MARKETS #

# Sort predictions from high to low
print("- Analyzing predictions")
new_attendance_df = new_attendance_df.sort_values(by="Predicted Attendance", ascending=False)
new_attendance_df.reset_index(inplace=True, drop=True)
new_attendance_df.index += 1

# Calculate place scores for predictions
place_dict = dict()
for i in range(1, len(new_attendance_df)):
    market = new_attendance_df["Market"][i]
    dome = new_attendance_df["Dome"][i]
    if market not in place_dict.keys():
        place_dict[market] = [0, 1]
        place_dict[market][dome] = i
    else:
        place_dict[market][dome] += i

# Create dataframe for place scores for predictions
place_list = []
for market in place_dict.keys():
    market_df = new_attendance_df[new_attendance_df["Market"] == market]
    attendance_sum_dome = sum(market_df[market_df["Dome"] == 1]["Predicted Attendance"])
    attendance_sum_open = sum(market_df[market_df["Dome"] == 0]["Predicted Attendance"])
    place_list.append((market, 1, place_dict[market][1], attendance_sum_dome))
    place_list.append((market, 0, place_dict[market][0], attendance_sum_open))

place_df = pd.DataFrame(place_list, columns=["Market", "Dome", "Place Score", "Predicted Total Attendance"])
place_df = place_df.sort_values(by="Place Score", ascending=True)
place_df = place_df.groupby("Market", as_index=False).first()
place_df = place_df.sort_values(by="Place Score", ascending=True)
place_df.reset_index(inplace=True, drop=True, names=[*range(1, len(place_df)+1)])
place_df.index += 1
place_df["Average Yearly Attendance"] = place_df["Predicted Total Attendance"]/10
place_df["Average Game Attendance"] = place_df["Average Yearly Attendance"]/81
place_df = place_df[["Market", "Dome", "Place Score", "Average Game Attendance"]]
dfi.export(place_df, f"{ROOT_PATH}/Visualizations/NewMarketPlaceScoreTable.png")

# Plot New Market vs. Avg. Game Attendance
print("- Plotting new data market vs. average game attendance")
plt.figure(figsize=(15, 10))
plt.bar(place_df["Market"], place_df["Average Game Attendance"])
plt.xlabel("Market")
plt.xticks(place_df["Market"], place_df["Market"], rotation=270)
plt.ylabel("Average Game Attendance")
plt.title("Potential New Market Predicted Average Game Attendance")
plt.savefig(f"{ROOT_PATH}/Visualizations/AttendanceXGBNewMarketPlot.png")

# ANALYTICS COMPARING CURRENT AND NEW MARKETS #

# Calculate place scores for predictions
print("- Comparing predictions to current market data")
full_place_dict = dict()
for i in range(1, len(full_attendance_df)):
    market = full_attendance_df["Market"][i]
    dome = full_attendance_df["Dome"][i]
    if market not in full_place_dict.keys():
        full_place_dict[market] = [0, 1]
        full_place_dict[market][dome] = i
    else:
        full_place_dict[market][dome] += i

# Create dataframe for place scores for predictions
full_place_list = []
for market in full_place_dict.keys():
    market_df = full_attendance_df[full_attendance_df["Market"] == market]
    attendance_sum_dome = sum(market_df[market_df["Dome"] == 1]["Attendance"])
    attendance_sum_open = sum(market_df[market_df["Dome"] == 0]["Attendance"])
    full_place_list.append((market, 1, full_place_dict[market][1], attendance_sum_dome))
    full_place_list.append((market, 0, full_place_dict[market][0], attendance_sum_open))

full_place_df = pd.DataFrame(full_place_list, columns=["Market", "Dome", "Place Score", "Total Attendance"])
full_place_df = full_place_df[full_place_df["Total Attendance"] != 0]
full_place_df = full_place_df.sort_values(by="Place Score", ascending=True)
full_place_df = full_place_df.groupby("Market", as_index=False).first()
full_place_df = full_place_df.sort_values(by="Place Score", ascending=True)
full_place_df.reset_index(inplace=True, drop=True)
full_place_df.index += 1
full_place_df["Average Yearly Attendance"] = full_place_df["Total Attendance"]/10
full_place_df["Average Game Attendance"] = round(full_place_df["Average Yearly Attendance"]/81, 0)
full_place_df = full_place_df[["Market", "Dome", "Place Score", "Average Game Attendance"]]
dfi.export(full_place_df[0:25], f"{ROOT_PATH}/Visualizations/FullMarketPlaceScoreTable.png")

# Plot Market vs. Avg. Game Attendance
print("- Plotting full data market vs. average game attendance")
plt.figure(figsize=(15, 10))
plt.bar(full_place_df["Market"], full_place_df["Average Game Attendance"])
plt.xlabel("Market")
plt.xticks(full_place_df["Market"], full_place_df["Market"], rotation=270)
plt.ylabel("Average Game Attendance")
plt.title("Current and New Market Average Game Attendance")
plt.savefig(f"{ROOT_PATH}/Visualizations/AttendanceXGBFullMarketPlot.png")
