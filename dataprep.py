import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import data_file, train_test_ratio, class_names, scaler

def prepared_data():
    train_df = pd.read_csv(data_file)
    train_df = train_df.drop(['trading_code', 'Unnamed: 0'], axis=1)
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df.set_index('date', inplace=True)
    train_df = train_df.sort_index()
    train_df = train_df.drop_duplicates()
    dataset = train_df.filter(class_names)
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, dataset

def multiVariant_timeseries_data_XY(data, window):
     X = []
     Y = []
     end = len(data)
     for i in range(window, end):
         X.append(data[i-window : i, :])
         Y.append(data[i, :])
     return np.array(X), np.array(Y)

def generate_train_dataset():
    dataset, _ = prepared_data()
    X, Y = multiVariant_timeseries_data_XY(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, shuffle=False)
    return (X_train, y_train), (X_test, y_test)

def generate_test_dataset():
    sc_data, dataset = prepared_data()
    training_data_len = math.ceil(len(sc_data) * .8)
    testing_data_len = math.ceil(len(dataset) * .05)
    valid_data = sc_data[training_data_len-100:,:]
    X_test = []
    y_test = dataset.iloc[training_data_len:-testing_data_len, :].values #not scaled data
    for i in range(100, len(valid_data)):
        X_test.append(valid_data[i-100:i,:]) #0 
    return (X_test, y_test)