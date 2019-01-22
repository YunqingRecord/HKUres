from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os
import numpy as np


def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')

'''
dataset = read_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\floors_2018_06_18\\11_F.csv',
                   parse_dates=['time'], index_col=1,date_parser=parse,
                   usecols=['time', 'location', 'total', 'ac', 'light', 'socket','other', 'mixed_usage',
                            'next_holiday','temperature_max', 'temperature_min'])

# dataset = DataFrame(dataset)
# dataset = dataset.pop('location')
# dataset.drop(['location'], axis=1, inplace=True)
# dataset.drop(['water_heater'], axis=1, inplace=True)
# dataset.drop(['cooking_appliance'], axis=1, inplace=True)

dataset.columns = ['location', 'total', 'ac', 'light', 'socket','other', 'mixed_usage',
                            'next_holiday','temperature_max', 'temperature_min']

# dataset.index.name = 'Date'
# dataset['No'].fillna(0, inplace=True)
# dataset['temp'].fillna(10, inplace=True)
# dataset = dataset[24:]

print(dataset.head(5))

dataset.to_csv('1.csv')


# Firstly I load the smoothed dataset

dataset = read_csv('1.csv', header=0, index_col=0)
dataset.columns =  ['location', 'total', 'ac', 'light', 'socket','other', 'mixed_usage',
                            'next_holiday','temperature_max', 'temperature_min']
values = dataset.values

# try to plot the exiting scatters except the win_dir
groups= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# fig = plt.figure()
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()
'''


# convert series to supervised learning(add_labels in time_series data),very important
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    combine = concat(cols, axis=1)
    combine.columns = names
    # drop rows with NaN values
    if dropnan:
        combine.dropna(inplace=True)
    return combine


# data1 = series_to_supervised(data=dataset)


def load_data(filename):  # add labels to the dataset

    dataset = read_csv(filename, index_col=0,
                       usecols=['time', 'total', 'ac', 'light', 'socket', 'other', 'mixed_usage',
                                'next_holiday', 'temperature_max', 'temperature_min'])
    values = dataset.values

    # encoder = LabelEncoder()
    # values[:, 0] = encoder.fit_transform(values[:, 0])

    values = values.astype('float32')
    # normalize features in the range of (0, 1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scale(values, axis=0, with_mean=True, with_std=True, copy=True)
    # scaled = scaler.fit_transform(values)
    # frame as supervised learning
    combine = series_to_supervised(scaled, 1, 1)  # future is one_step
    # drop columns overload
    # combine.drop(combine.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

    values = combine.values[:, :10]
    return values


def read_file(first_path = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\floors_2018_06_18\\'):
    # read all files iteratively and process into new csv file
    csv_list = os.listdir(first_path)
    i = 0
    for csv_file in csv_list:
        i += 1
        filename = first_path + csv_file
        values = DataFrame(load_data(filename))
        values.to_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\'+'addlabel\\' + str(i) + '.csv')


def partition(values): # used to divide the train and test data
    # values = load_data(filename)
    num_trained = 700
    num_valid = 122

    train_set = values[0: num_trained, :]
    valid_set = values[num_trained:num_trained+num_valid, :]
    test_set = values[num_trained+num_valid:, :]

    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]

    x_valid = valid_set[:, :-1]
    y_valid = valid_set[:, -1]

    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    y_test_final = test_set[:, -1]
    # reshape to teh [samples, time_steps, num_of_features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    # y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))

    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # y_test  = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))

    return x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_final


