from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime


def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')


dataset = read_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\floors_2018_06_18\\11_F.csv',
                   parse_dates=['time'], index_col=1, date_parser=parse)

dataset = DataFrame(dataset)
dataset.drop(['location']) # , axis=1, inplace=True)
dataset.drop('fridge', axis=4, inplace=True)
dataset.drop('water_heater', axis=5, inplace=True)
dataset.drop('cooking_appliance', axis=6, inplace=True)

dataset.columns = ['total', 'ac', 'light', 'socket',
                   'other', 'mixed_usage', 'next_holiday', 'temperature_max', 'temperature_min']

# dataset.index.name = 'Date'
# dataset['No'].fillna(0, inplace=True)
# dataset['temp'].fillna(10, inplace=True)
dataset = dataset[24:]

print(dataset.head(5))

dataset.to_csv('1.csv')


# Firstly I load the smoothed dataset

dataset = read_csv('1.csv', header=0, index_col=0)
dataset.columns = ['total', 'ac', 'light', 'socket',
                   'other', 'mixed_usage', 'next_holiday', 'temperature_max', 'temperature_min']
values = dataset.values

# try to plot the exiting scatters except the win_dir
groups= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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


def load_data(path='Peking_air_pollution.csv'):
    dataset = read_csv(path, header=0, index_col=0)
    values = dataset.values

    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])

    values = values.astype('float32')
    # normalize features in the range of (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    combine = series_to_supervised(scaled, 1, 1)  # either past or future is one_step
    # drop columns overload
    combine.drop(combine.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

    values = combine.values
    num_trained = 365*2*24
    num_valid = 365*24

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

    x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # y_test  = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))

    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler, y_test_final