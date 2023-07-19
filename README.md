# Predicting MLB Attendance Based on Housing Income and Valuation Data
Team 116:
<br />Joshua Mayanja, Warren Spann, Anthony Palmieri, Yutai Liu, Jon Martin

## Description
This project attempts to predict yearly attendance for MLB markets based on housing and income data from the US census between the years of 2011-2021. The year 2020 was removed, as no fans were allowed at MLB games due to the COVID-19 pandemic.

All code can be found in the Final Code folder of this project.

## Data
All .py files in this section can be found in `Code/DataCleaning` and all csv data can be found in `Data/CSVData`.

Three data sources were used for this project. One, `us_income_zipcode.csv` was a csv file of housing and income data from the US Census (Note that this file is in a zip folder due to the large file limit and will be extracted when `clean_data.py` is run). 

The data came in zip codes, so each zip code had to be collected and grouped in to markets, done by `find_zip_codes.py`. `find_attendance.py` takes the zip code data and maps the corresponding yearly attendance values. `find_valuations.py` does the same for yearly valuation data. Note that the attendance and valuation data was scraped from baseball-reference.com and wikipedia/forbes.com respectively, and when the script is run, it can only take in data for a single market at a time. 

This data is then all put together with `clean_data.py` which creates two csv files `cleaned_current_markets_data.csv` and `cleaned_new_markets.csv`. These csv files contain all attendance, valuation, and housing/income data (the former for the current markets, and the latter for the potential new markets).

Finally `split_data.py` splits these two csv files into a training and a test set using a rotation. 70% of the data went to the training set, and the remaining 30% to the test set. The training and test sets can be found in `Data/SplitData`. 

## Location Clustering
Code Description

## Attendance Clustering
All data in this section can be found in `Models/Clustering`.

The file `Clustering Market by Attendance.R` creates the cluster markets by attendance, using K-means clustering and the elbow method and produces plots for visualisation of the cluster markets

## Linear Regression Model
Code Description

## XGBoost Model
All data in this section can be found in `Models/Attendance/XGBoost`.

The main file is `attendance_xgboost.py` which creates an XGBoost model. The script performs feature selection, cross validation to find the best model, and then trains a final model. It provides plots and tables for analysis in `Visualizatoins`, as well as `AttendanceXGBEval.txt` providing initial model stats. The model is then saved as `AttendanceXGBModel.json`.

Predictions using this model for new market yearly attendance are made in `predict_attendance_xgboost.py`, and saved in `Data/Predictions/PredictedNewMarketAttendanceXGBoost.csv`.

Analysis is performed in `attendance_xgboost_analysis.py` which creates tables and graphs (stored in `Visualizations`) that analyze predicted new market performance. Also calculates place scores, which are the sum of the placing of a market's attendance each year (ex, if a market placed first in attendance all 10 years, it's place score would be 10).

## Deep Learning Model
Code Description
