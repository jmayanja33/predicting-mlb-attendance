import yaml
import json
from yaml import CLoader as Loader
from root_path import ROOT_PATH

# Change `MARKET`, `YEAR`, and `VALUATION` variables to corresponding values
MARKET = "Test"
YEAR = 2011
VALUATION = 360 * 1000000

# Read yml file into dictionary
READ_FILE = open(f"{ROOT_PATH}/Data/Valuations/current_market_valuation.yml", "r")
valuations = yaml.load(READ_FILE, Loader)
READ_FILE.close()
if valuations is None:
    valuations = dict()

default_dict = {2021: 0, 2020: 0, 2019: 0, 2018: 0, 2017: 0, 2016: 0, 2015: 0,
                2014: 0, 2013: 0, 2012: 0, 2011: 0}

# Write new attendance numbers into yml and json file
if MARKET not in valuations.keys():
    valuations[MARKET] = default_dict
valuations[MARKET][YEAR] = VALUATION


# Write results to yml and json files
WRITE_FILE_YML = open(f"{ROOT_PATH}/Data/Valuations/current_market_valuation.yml", "w")
WRITE_FILE_JSON = open(f"{ROOT_PATH}/Data/Valuations/current_market_valuation.json", "w")
yaml.dump(valuations, WRITE_FILE_YML)
json.dump(valuations, WRITE_FILE_JSON)
WRITE_FILE_YML.close()
WRITE_FILE_JSON.close()
