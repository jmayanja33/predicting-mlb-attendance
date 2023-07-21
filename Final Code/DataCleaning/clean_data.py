import pandas as pd
import yaml
from yaml import CLoader as Loader
from tqdm import tqdm
from zipfile import ZipFile
from root_path import ROOT_PATH

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Initialize dictionary to reference if a market has a domed stadium
domes = {
    "Milwaukee": 1, "Los Angeles": 0, "St. Louis": 0, "Phoenix": 0, "New York": 0, "Philadelphia": 0, "Detroit": 0,
    "Denver": 0, "Boston": 0, "Dallas": 1, "Cincinnati": 0, "Chicago": 0, "Kansas City": 0, "Miami": 1, "Houston": 1,
    "Washington DC": 0, "San Francisco Bay Area": 0, "Baltimore": 0, "San Diego": 0, "Pittsburgh": 0, "Cleveland": 0,
    "Seattle": 1, "Minneapolis-St.Paul": 0, "Tampa Bay": 1, "Atlanta": 0
}

#  Initialize dictionary to reference how many teams a market has
num_teams = {
    "Milwaukee": 1, "Los Angeles": 2, "St. Louis": 1, "Phoenix": 1, "New York": 2, "Philadelphia": 1, "Detroit": 1,
    "Denver": 1, "Boston": 1, "Dallas": 1, "Cincinnati": 1, "Chicago": 2, "Kansas City": 1, "Miami": 1, "Houston": 1,
    "Washington DC": 1, "San Francisco Bay Area": 2, "Baltimore": 1, "San Diego": 1, "Pittsburgh": 1, "Cleveland": 1,
    "Seattle": 1, "Minneapolis-St.Paul": 1, "Tampa Bay": 1, "Atlanta": 1
}


def format_zip_codes(df):
    """Function which turns zip codes into strings and adds leading zeros"""
    df["ZIP"] = [str(i) for i in df["ZIP"]]
    print("- Formatting ZIP Codes:")
    for i in tqdm(range(len(df))):
        zipcode = df["ZIP"][i]
        while len(zipcode) < 5:
            zipcode = "0" + zipcode
            df["ZIP"][i] = zipcode


def compress_market_zips(df, market_dict, market_type):
    """Function to take all zip codes for a market and compress them into one data point by year"""
    markets = market_dict.keys()
    print(f"- Compressing yearly data for all market ZIP Codes")
    compressed_markets = []
    data_cols = [col for col in df.columns if col not in {"ZIP", "Year", "Geography", "Geographic Area Name"}]
    if market_type == "current":
        headers = ["Market", "Year", "Dome", "NumTeams", "Attendance", "Valuation"]
    else:
        headers = ["Market", "Year", "Dome", "NumTeams"]
    # Iterate through markets
    for market in tqdm(markets):
        market_df = df[df["ZIP"].isin(market_dict[market])]
        years = market_df["Year"].unique()
        # Iterate through years
        for year in years:
            year_df = market_df[market_df["Year"] == year][data_cols]
            # Case for current markets, find stadium type, attendance
            if market_type == "current":
                compressed_market = [market, year, domes[market], num_teams[market], attendances[market][year],
                                     valuations[market][year]]
                # Find sum of each column
                for col in data_cols:
                    compressed_market.append(year_df[col].sum(skipna=True))
                compressed_markets.append(compressed_market)
            # Case for new markets, test both stadium types
            else:
                compressed_market_open = [market, year, 0, 1]
                compressed_market_dome = [market, year, 1, 1]
                # Find sum of each column
                for col in data_cols:
                    compressed_market_open.append(year_df[col].sum(skipna=True))
                    compressed_market_dome.append(year_df[col].sum(skipna=True))
                compressed_markets.append(compressed_market_open)
                compressed_markets.append(compressed_market_dome)
    # Complete list of headers
    for col in data_cols:
        headers.append(col)
    # Return cleaned data
    return compressed_markets, headers


def create_csv_file(compressed_markets, headers, output_filename):
    """Function which saves cleaned data as a csv file"""
    print(f"- Saving cleaned data to file: {output_filename}")
    market_df = pd.DataFrame(data=compressed_markets, columns=headers)
    market_df.to_csv(f"{ROOT_PATH}/Data/CSVData/{output_filename}", index=False)


def clean_data(df, markets, output_filename, market_type):
    """Function which filters data for all given zipcodes and saves it as a csv file"""
    compressed_markets, headers = compress_market_zips(df, markets, market_type)
    create_csv_file(compressed_markets, headers, output_filename)


def load_census_data():
    """Function to extract census dataset from zip as it was too large for git"""
    with ZipFile(f"{ROOT_PATH}/Data/CSVData/us_income_zipcode.csv.zip", 'r') as zip_file:
        zip_file.extract("us_income_zipcode.csv", f"{ROOT_PATH}/Data/CSVData/us_income_zipcode.csv")
        zip_file.close()
    data = pd.read_csv(f"{ROOT_PATH}/Data/CSVData/us_income_zipcode.csv/us_income_zipcode.csv")
    return data


# Load in data
print("PROGRESS: ")
print("\n- Loading in full income dataset")
data = load_census_data()
attendance_file = open(f"{ROOT_PATH}/Data/Attendance/current_market_attendance.yml", "r")
valuation_file = open(f"{ROOT_PATH}/Data/Valuations/current_market_valuation.yml", "r")
attendances = yaml.load(attendance_file, Loader)
valuations = yaml.load(valuation_file, Loader)
format_zip_codes(data)

# Load current market zip codes into dictionary
print("\n- Cleaning current market data")
current_market_file = open(f"{ROOT_PATH}/Data/ZipCodes/CurrentMarkets/current_market_zip_codes.yml", "r")
current_markets = yaml.load(current_market_file, Loader)
clean_data(data, current_markets, "cleaned_current_markets_data.csv", market_type="current")


# Load potential new market zip codes into dictionary
print("- Cleaning new market data")
new_market_file = open(f"{ROOT_PATH}/Data/ZipCodes/NewMarkets/new_market_zip_codes.yml", "r")
new_markets = yaml.load(new_market_file, Loader)
clean_data(data, new_markets, "cleaned_new_markets_data.csv", market_type="new")
