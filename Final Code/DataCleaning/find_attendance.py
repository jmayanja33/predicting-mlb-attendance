import yaml
import json
from yaml import CLoader as Loader
from root_path import ROOT_PATH

# Change `MARKET`, `YEAR`, and `ATTENDANCE` variables to corresponding values
MARKET = "Washington DC"
YEAR = 2011
ATTENDANCE = 1940478

# Read yml file into dictionary
READ_FILE = open(f"{ROOT_PATH}/Data/Attendance/current_market_attendance.yml", "r")
attendances = yaml.load(READ_FILE, Loader)
READ_FILE.close()

# Default values for each market
default_dict = {2021: 0, 2020: 0, 2019: 0, 2018: 0, 2017: 0, 2016: 0, 2015: 0,
                2014: 0, 2013: 0, 2012: 0, 2011: 0}

# Write new attendance numbers into yml and json file
if MARKET not in attendances.keys():
    attendances[MARKET] = default_dict
attendances[MARKET][YEAR] = ATTENDANCE
WRITE_FILE_YML = open(f"{ROOT_PATH}/Data/Attendance/current_market_attendance.yml", "w")
WRITE_FILE_JSON = open(f"{ROOT_PATH}/Data/Attendance/current_market_attendance.json", "w")
yaml.dump(attendances, WRITE_FILE_YML)
json.dump(attendances, WRITE_FILE_JSON)
WRITE_FILE_YML.close()
WRITE_FILE_JSON.close()
