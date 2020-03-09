import datetime, json
import numpy as np
from data_preprocess import (convert_entry_df, create_holiday_feature, add_features_datetime_YMD, 
                             time_period_bin, own_labelencode, load_run_mlmodel)

Temperature = 123
Weather_main = "Drizzle"
Date_Time = datetime.datetime.now()

x_pred = convert_entry_df(temp=Temperature, weather_main=Weather_main, date_time=Date_Time)
x_pred = create_holiday_feature(dataset=x_pred, column='date_time', days=3)
x_pred = add_features_datetime_YMD(x_pred, column="date_time", feature_name=['day', 'time'])
x_pred = time_period_bin(x_pred, 'time_period')
dataset_labelencode_weather = json.load(open('dataset_labelencode_weather.txt'))
dataset_labelencode_day = json.load(open('dataset_labelencode_day.txt'))
dataset_labelencode_time = json.load(open('dataset_labelencode_time.txt'))
x_pred = own_labelencode(x_pred, dataset_labelencode_weather, 'weather_main')
x_pred = own_labelencode(x_pred, dataset_labelencode_day, 'day_of_the_week')
x_pred = own_labelencode(x_pred, dataset_labelencode_time, 'time_period')
x_pred = x_pred.iloc[:, :].values
y_pred = load_run_mlmodel('finalized_model.pkl', x_pred)[0]