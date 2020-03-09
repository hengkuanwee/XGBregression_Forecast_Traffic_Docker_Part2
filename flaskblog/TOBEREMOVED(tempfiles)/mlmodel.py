# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:27:13 2020

@author: kuanw
"""
# System
import io, os, sys, datetime, math, calendar, time
import holidays
# Data Manipulation
import numpy as np
import pandas as pd

# Data Preprocessing
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score,f1_score,r2_score,explained_variance_score
from xgboost import XGBClassifier, XGBRegressor

# MySQL
import mysql.connector
from mysql.connector import Error

# Import Data from MySQL "aiap_traffic" database
def MySQL_Connection(host, mysql_user, mysql_password, port, database, table):
	config = {
		'host': host,
		'user': mysql_user,
		'password': mysql_password,
		'port': port,
		'database': database
	}
	
	tries = 30
	while tries > 0:
		try:
			connection = mysql.connector.connect(**config)
			if connection.is_connected():
				db_Info = connection.get_server_info()
				print("\nConnected to MySQL Server version ", db_Info)
				dataset = pd.read_sql("SELECT * FROM " + table, con=connection)
				print(database + " is extracted." )   
				break  
		except:
			print("Waiting for MySQL database to be set up...")
			time.sleep(10)
		finally:
			tries -= 1
	
	if connection.is_connected():
		return dataset
		connection.close()
		print("MySQL connection is closed")
		print("Done")
	else:
		print("MySQL connection failed")

# Drop features
def drop_features (dataset, column, axis=1):
	dataset = dataset.drop(columns=[column], axis=axis)
	
	return dataset
	
# Create the additional features "hours_before_holiday" and "hours_after_holiday", and remove "holiday"
def add_features_holiday (dataset, column, hours=24):
	# Create blank list for hours_before and hours_after, if the row is within 24 hours from a holiday, we will append the row number to it
	hours_before = []
	hours_after = []
	# Create blank list for hours_holiday, if the row is the holiday itself, we will append the row number to it
	hours_holiday = []
	# Create numpy arrays of False, if row number is within 24 hours from a holiday, we will change it to True
	before_holiday = np.zeros(len(dataset[column]))
	after_holiday = np.zeros(len(dataset[column]))
	for index, row in dataset[column].to_frame().iterrows():
		# If there is a holiday, append the relevant number to hours_holiday
		if row[column] != "None":
			hours_holiday.append(index)
	
	# Append the relevant row humbers to hours_before and hours_after
	for i in hours_holiday:
		for hour in range(0, hours+1):
			hours_before.append(i - hour)
			hours_after.append(i + hour)
			
	# Remove the row rumbers that are out of range
	hours_before = np.asarray(hours_before)
	hours_before = hours_before[(hours_before>=0) & (hours_before<=len(dataset[column]))]
	hours_after = np.asarray(hours_after)
	hours_after = hours_after[(hours_after>=0) & (hours_after<=len(dataset[column]))]
	
	# Change numpy array to true, if the respective row number within 24 hours from a holiday
	before_holiday[hours_before.tolist()] = 1
	after_holiday[hours_after.tolist()] = 1
	
	# Convert hours_before_holiday and hours_after_holiday to dataframe and merge to original dataset
	before_holiday = pd.DataFrame(before_holiday,  columns=['before_holiday'])
	after_holiday = pd.DataFrame(after_holiday,  columns=['after_holiday'])
	
	dataset =  pd.concat([dataset, before_holiday], axis=1, sort=False)
	dataset =  pd.concat([dataset, after_holiday], axis=1, sort=False)
	
	# Drop column as relevant features were already extracted and feature takes into account column
	dataset = dataset.drop([column], axis = 1)
			
	return dataset

def convert_datetime_format(dataset, column):
	dataset[column] = pd.to_datetime(dataset[column], format="%Y-%m-%d %H:%M:%S")
	
	return dataset

# Create the additional features "year", "month", "day_of_the_week" and "time_period", and remove "date_time"
def add_features_datetime_YMD (dataset, column="date_time", feature_name=["year", "month", "day", "time"]):
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

# Classify time period into bins of Morning, Afternoon, Evening and Night. For each bin, the traffic is expected to be different
def time_period_bin(dataset, column):
	dataset[column] = pd.cut(dataset[column], 
									bins=[0,6,12,18,23], 
									labels=['Night','Morning','Afternoon','Evening'],
									include_lowest=True)
	return dataset

# Encoding categorical data (i.e. Creating dummy variables)
# Label encode each categorical variable
def labelencode (dataset, columns = [0]):
	for column in columns:
		labelencoder_data = LabelEncoder()
		dataset[:,column] = labelencoder_data.fit_transform(dataset[:,column])
	return dataset

# Gridsearch to find the best model parameters
def XGBRegressor_gridsearch(X_train, y_train, X_test, y_test):
	xgb_model = XGBRegressor(objective='reg:squarederror', 
							  tree_method='exact', 
							  early_stopping_rounds = 50)
							  
	parameters = {'max_depth': [9], 
					'learning_rate': [0.015], 
					'n_estimators': [1500], 
					'gamma': [0], 
					'min_child_weight': [1], 
					'subsample': [0.8], 
					'colsample_bytree': [0.9], 
					#'seed': [10]
				 }
				 
	clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, 
					   cv=10,
					   verbose=0, refit=True)
	
	return clf.fit(X_train, y_train)
	
'''
	print("Done")
	
	# Print model report:
	print ("\n"+ "\033[1m" + "Model Report" + "\033[0m")
	print("Best: Accuracy of %f using: \n %s" % (clf.best_score_, clf.best_params_))
	
	# Predicting the test set results
	y_pred = clf.predict(X_test).astype(int)
	
	# Print prediction report on test set
	print ("\n" + "\033[1m"+ "Prediction Report" + "\033[0m")
	print("Adj R-squared : " + str(r2_score(y_test,y_pred)))
	print("Variance: " + str(explained_variance_score(y_test,y_pred)))
	
	# Print predictions
	predictions = np.concatenate([(np.squeeze(y_test), y_pred)]).T
	predictions = pd.DataFrame(predictions, columns=["traffic_volume_test", "traffic_volume_pred"])
	
	print ("\n" + "\033[1m"+ "First 30 predictions" + "\033[0m")
	print(predictions.head(30))
'''


dataset = MySQL_Connection('localhost', 'root', 'gftyo32HKW', 3306, 'aiap_traffic', 'traffic_data')

dataset = drop_features(dataset, "snow_1h", 1)
dataset = drop_features(dataset, "rain_1h", 1)
dataset = drop_features(dataset, "clouds_all", 1)
dataset = drop_features(dataset, "weather_description", 1)

dataset = add_features_holiday(dataset, "holiday")

dataset = convert_datetime_format(dataset, "date_time")

dataset = add_features_datetime_YMD (dataset, column="date_time", feature_name=["day", "time"])

dataset = time_period_bin(dataset, "time_period")

X = dataset.drop(["traffic_volume"], axis = 1).iloc[:, :].values
y = dataset.loc[:,"traffic_volume"].values

X = labelencode(dataset=X, columns=[1,2,3,4,5] )
print(pd.DataFrame(X))
ct_1 = ColumnTransformer(
		[('one_hot_encoder', OneHotEncoder(drop='first', categories='auto'), [1,2,3,4,5])],
		remainder = 'passthrough')
X = ct_1.fit_transform(X).toarray()
	
	# Splitting the dataset into the Training set and Test set
print ("Step 9 - Split into Training set and Test set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	
	# Conduct GridSearch, to find the optimal hyper-parameters
print ("\n" + "\033[1m"+ "Training XGBRegressor Model..." + "\033[0m")
clf = XGBRegressor_gridsearch(X_train, y_train, X_test, y_test)

# Ask for user input - date, temp, weather_description
def add_data ():
    total_entry = []
    data_rows = abs(int(input("please let us know the number of data to input: ")))
    for i in range(data_rows):
        date_time =str(input("please enter the date: "))
        temp = float(input("please input the temperature: "))
        weather_description = str(input("please input the weather description: "))
        single_entry = [date_time, temp, weather_description]
        total_entry.append(single_entry)
    
    #x_pred = pd.DataFrame(columns=['date_time','weather_description', 'temp'])
    final_entry = pd.DataFrame(total_entry, columns=['date_time','weather_description', 'temp'] )
    return final_entry

x_pred = add_data()

def create_holiday_feature1 (dataset):
    sg_holidays = holidays.Singapore()
    dataset['holiday'] = [
            sg_holidays.get(str(i)) if i in sg_holidays else "None" for i in dataset['date_time']
            ]
    return dataset

# Ask for user input - date, temp, weather_description
def add_single_data ():
    #request for user input and convert into dataframe
    date_time =str(input("please enter the date: "))
    temp = float(input("please input the temperature: "))
    weather_main = str(input("please input the weather main: "))
    single_entry = [temp, weather_main, date_time]    
    final_entry = pd.DataFrame([single_entry], columns=['temp', 'weather_main', 'date_time'] )

    #pre-process datatype
    final_entry = convert_datetime_format(final_entry, "date_time")
    return final_entry

x_pred = add_single_data()

# Pre-process the data
# Create holiday features


def create_holiday_feature (dataset, column, days=3):
    sg_holidays = holidays.Singapore(years=dataset.iloc[0][column].year)
    isit_before_holiday = False
    isit_after_holiday = True
    
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

x_pred = create_holiday_feature (x_pred, 'date_time', 3)

# Create date_time features
x_pred = add_features_datetime_YMD (x_pred, column="date_time", feature_name=["day", "time"])
x_pred = time_period_bin(x_pred, "time_period")

# Encoding categorical data
def labelencode_dictionary (dataset, column):
    extracted_column = np.sort(dataset[column].unique())
    label_dict = {}
    label_number = 0
    for i in extracted_column:
        label_dict[i] = label_number
        label_number += 1
    return label_dict

dataset_labelencode_weather = labelencode_dictionary(dataset, 'weather_main')
dataset_labelencode_day = labelencode_dictionary(dataset, 'day_of_the_week')
dataset_labelencode_time = labelencode_dictionary(dataset, 'time_period')

def own_labelencode (dataset, labelencode_dictionary, column):
    dataset[column] = labelencode_dictionary[dataset.iloc[0][column]]
    return dataset

x_pred1 = x_pred
x_pred1 = own_labelencode(x_pred1, dataset_labelencode_weather, 'weather_main')
x_pred1 = own_labelencode(x_pred1, dataset_labelencode_day, 'day_of_the_week')
x_pred1 = own_labelencode(x_pred1, dataset_labelencode_time, 'time_period')
x_pred1 = x_pred1.iloc[:,:].values

# ORIGINAL DATASET
dataset = own_labelencode(dataset, dataset_labelencode_weather, 'weather_main')
dataset = own_labelencode(dataset, dataset_labelencode_day, 'day_of_the_week')
dataset = own_labelencode(dataset, dataset_labelencode_time, 'time_period')

X = dataset.drop(["traffic_volume"], axis = 1).iloc[:, :].values
y = dataset.loc[:,"traffic_volume"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

xgb_model = XGBRegressor(objective='reg:squarederror', 
							  tree_method='exact', 
							  early_stopping_rounds = 50)
							  
parameters = {'max_depth': [9], 
			  'learning_rate': [0.015], 
			  'n_estimators': [1500], 
		 	  'gamma': [0], 
			  'min_child_weight': [1], 
			  'subsample': [0.8], 
			  'colsample_bytree': [0.9], 
			  #'seed': [10]
			}
				 
clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, 
					   cv=10,
					   verbose=0, refit=True)
	
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).astype(int)
predictions = np.concatenate([(np.squeeze(y_test), y_pred)]).T
predictions = pd.DataFrame(predictions, columns=["traffic_volume_test", "traffic_volume_pred"])
	
y_pred1 = clf.predict(x_pred1).astype(int)

##### COMPILE ALL FOR THE ORIGINAL DATASET
dataset = MySQL_Connection('localhost', 'root', 'gftyo32HKW', 3306, 'aiap_traffic', 'traffic_data')

dataset = drop_features(dataset, "snow_1h", 1)
dataset = drop_features(dataset, "rain_1h", 1)
dataset = drop_features(dataset, "clouds_all", 1)
dataset = drop_features(dataset, "weather_description", 1)
dataset = drop_features(dataset, "holiday", 1)

dataset = convert_datetime_format(dataset, "date_time")

import holidays
from datetime import timedelta
dataset = create_holiday_feature (dataset, 'date_time', 3)
dataset = add_features_datetime_YMD (dataset, column="date_time", feature_name=["day", "time"])
dataset = time_period_bin(dataset, "time_period")

# CREATE LABEL ENCODE DICTIONARY
def labelencode_dictionary (dataset, column):
    extracted_column = np.sort(dataset[column].unique())
    label_dict = {}
    label_number = 0
    for i in extracted_column:
        label_dict[i] = label_number
        label_number += 1
    return label_dict
dataset_labelencode_weather = labelencode_dictionary(dataset, 'weather_main')
dataset_labelencode_day = labelencode_dictionary(dataset, 'day_of_the_week')
dataset_labelencode_time = labelencode_dictionary(dataset, 'time_period')

# EXPORT LABEL ENCODE DICTIONARY
import json
json.dump(dataset_labelencode_weather, open('dataset_labelencode_weather.txt','w'))
json.dump(dataset_labelencode_day, open('dataset_labelencode_day.txt','w'))
json.dump(dataset_labelencode_time, open('dataset_labelencode_time.txt','w'))
TESTINGFORDICTIONARY = json.load(open("dataset_labelencode_weather.txt"))
print(TESTINGFORDICTIONARY)

# ORIGINAL DATASET
def own_labelencode (dataset, labelencode_dictionary, column):
    dataset[column] = labelencode_dictionary[dataset.iloc[0][column]]
    return dataset

dataset = own_labelencode(dataset, dataset_labelencode_weather, 'weather_main')
dataset = own_labelencode(dataset, dataset_labelencode_day, 'day_of_the_week')
dataset = own_labelencode(dataset, dataset_labelencode_time, 'time_period')

# ML Model
X = dataset.drop(["traffic_volume"], axis = 1).iloc[:, :].values
y = dataset.loc[:,"traffic_volume"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

xgb_model = XGBRegressor(objective='reg:squarederror', 
                          tree_method='exact', 
                          early_stopping_rounds = 50)
                          
parameters = {'max_depth': [9],
              'learning_rate': [0.015],
              'n_estimators': [1500], 
              'gamma': [0],
              'min_child_weight': [1],
              'subsample': [0.8], 
              'colsample_bytree': [0.9], 
              #'seed': [10]
             }
             
clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, 
                   cv=10,
                   verbose=0, refit=True)

clf.fit(X_train, y_train)


# Print model report:
print ("\n"+ "\033[1m" + "Model Report" + "\033[0m")
print("Best: Accuracy of %f using: \n %s" % (clf.best_score_, clf.best_params_))

y_pred = clf.predict(X_test).astype(int)

# Print prediction report on test set
print ("\n" + "\033[1m"+ "Prediction Report" + "\033[0m")
print("Adj R-squared : " + str(r2_score(y_test,y_pred)))
print("Variance: " + str(explained_variance_score(y_test,y_pred)))

# Print predictions
predictions = np.concatenate([(np.squeeze(y_test), y_pred)]).T
predictions = pd.DataFrame(predictions, columns=["traffic_volume_test", "traffic_volume_pred"])

print ("\n" + "\033[1m"+ "First 30 predictions" + "\033[0m")
print(predictions.head(30))

print(os.getcwd())
print(os.chdir(r"."))
print(os.getcwd())


import joblib
# save the model to disk
filename = 'finalized_model.pkl'
joblib.dump(clf.best_estimator_, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.predict(X_test).astype(int)
print(result)


