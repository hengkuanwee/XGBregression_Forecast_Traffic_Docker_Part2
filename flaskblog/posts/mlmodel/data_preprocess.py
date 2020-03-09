# System imports
import numpy as np
import pandas as pd
import holidays
import datetime
from datetime import timedelta
import joblib
import json

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor

def convert_entry_df(temp, weather_main, date_time):
    entry = [temp, weather_main, date_time]    
    entry = pd.DataFrame([entry], columns=[ 'temp', 'weather_main', 'date_time'] )
    return entry

def create_holiday_feature(dataset, column, days=3):
    sg_holidays = holidays.Singapore(years=dataset.iloc[0][column].year)
    
    for i in range(days+1):
        if (dataset.iloc[0][column] + timedelta(days=i)) in sg_holidays:
            dataset['before_holiday'] = 1
            break
        else:
            dataset['before_holiday' ] = 0
    for i in range(days+1):
        if (dataset.iloc[0][column] - timedelta(days=i)) in sg_holidays:
            dataset['after_holiday'] = 1
            break
        else:
            dataset['after_holiday' ] = 0
    return dataset

def add_features_datetime_YMD(dataset, column="date_time", feature_name=["year", "month", "day", "time"]):
    # Create numpy arrays of zeros/empty string, we will replace the values subsequently
    dt_year = np.ones(len(dataset[column]))
    dt_month = np.ones(len(dataset[column]))
    dt_day = []
    dt_time = np.ones(len(dataset[column]))
    
    # Extract the relevant feature from column and update the features to dataset
    for feature in feature_name:
        if feature == "year":
            for index, row in dataset[column].to_frame().iterrows():
                dt_year[index] = row[column].year
            dt_year = pd.DataFrame(data=dt_year, columns=['year'], dtype=np.int64)
            dataset =  pd.concat([dataset, dt_year], axis=1, sort=False)
        elif feature == "month":
            for index, row in dataset[column].to_frame().iterrows():
                dt_month[index] = row[column].month
            dt_month = pd.DataFrame(data=dt_month, columns=['month'], dtype=np.int64)
            dataset =  pd.concat([dataset, dt_month], axis=1, sort=False)
        elif feature == "day":
            for index, row in dataset[column].to_frame().iterrows():
                dt_day.append(row[column].strftime('%A'))
            dt_day = pd.DataFrame(data=dt_day, columns=['day_of_the_week'], dtype=str)
            dataset =  pd.concat([dataset, dt_day], axis=1, sort=False)
        elif feature == "time":
            for index, row in dataset[column].to_frame().iterrows():
                dt_time[index] = row[column].hour
            dt_time = pd.DataFrame(data=dt_time, columns=['time_period'], dtype=np.int64)
            dataset =  pd.concat([dataset, dt_time], axis=1, sort=False)
  	# Drop column as relevant features were already extracted
    dataset = dataset.drop([column], axis = 1)
    return dataset

def time_period_bin(dataset, column):
    dataset[column] = pd.cut(dataset[column], 
                      bins=[0,6,12,18,23], 
                      labels=['Night','Morning','Afternoon','Evening'],
                      include_lowest=True)
    return dataset

def own_labelencode(dataset, labelencode_dictionary, column):
    dataset[column] = labelencode_dictionary[dataset.iloc[0][column]]
    return dataset

def load_run_mlmodel(saved_model, x_pred):
    filename = saved_model
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(x_pred).astype(int)
    return y_pred

