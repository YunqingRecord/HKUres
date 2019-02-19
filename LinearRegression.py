from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Transform_to_Training import scale_data, read_file, get_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pandas import concat, read_csv
import numpy as np
from numpy import concatenate
from math import sqrt
from pandas import DataFrame

'''
In this file, I use the LinearRegression Model to fit the data, predicting the next_day consumption,
by given the last day parameters.
 
'''


def read_file2(filepath = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\One_step_origin_with_label\\'):
    # read all added features files iteratively and process into new csv file
    # csv_list = os.listdir(filepath)
    # i = 0
    # for csv_file in csv_list:
    #     i += 1
    #     filename = filepath + csv_file
    #     dataset = read_csv(filename, usecols=['total', 'ac', 'light', 'socket', 'next_holiday', 'temperature_max',
    #                                           'temperature_min', 'pressure', 'dew_temp', 'cold_season'])
    #     return dataset  # Using the only first one data temporary
    filename = filepath + '11_F.csv'
    dataset = read_csv(filename, usecols=['total', 'ac', 'light', 'socket', 'next_holiday', 'temperature_max',
                                          'temperature_min', 'pressure', 'dew_temp', 'cold_season'])
    return dataset


def scale_data2():
    data = read_file2()
    values = data.values
    scalar = StandardScaler()
    scaled = scalar.fit_transform(values)
    return scaled, scalar


def series_to_supervised(values, n_in, n_out, dropnan=True, verbose=True):
    """
    :param: values: dataset scaled values
    :param: n_in: number of time lags (intervals) to use in each neurons
    :param: n_out: number of time-steps in future to predict
    :param: dropnan: whether to drop rows with NaN values after conversion to supervised learning
    :param: col_names: name of columns for dataset
    :param: verbose: whether to output some debug data
    :return: supervised
    """
    col_names = ['total', 'ac', 'light', 'socket', 'next_holiday',
                 'temperature_max', 'temperature_min', 'pressure', 'dew_temp', 'cold_season']
    n_vars = 1 if type(values) is list else values.shape[1]
    # if col_names is None: col_names = ["var%d" % (j + 1) for j in range(n_vars)]

    df = DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # col = list(shifted)
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together as a whole
    whole = concat(cols, axis=1)
    whole.columns = names

    # drop rows with NaN values
    if dropnan:
        whole.dropna(inplace=True)

    if verbose:
        print("\nsupervised data shape:", whole.shape)
    return whole


def partition(whole, time_steps=7, num_features=10):  # used to divide the train and test data
    values = whole.values
    num_trained = 720
    num_valid = 0

    train_set = values[0: num_trained, :]
    # valid_set = values[num_trained:num_trained+num_valid, :]
    test_set = values[num_trained+num_valid-26:, :]

    x_train = train_set[:, :time_steps*num_features]
    y_train = train_set[:, -num_features]

    # x_valid = valid_set[:, :time_steps*num_features]
    # y_valid = valid_set[:, time_steps*num_features]

    x_test = test_set[:, :time_steps*num_features]
    y_test = test_set[:, -num_features]

    # reshape to teh [samples, time_steps, num_of_features]
    # x_train = x_train.reshape((x_train.shape[0], time_steps, num_features))
    # # y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    # x_valid = x_valid.reshape((x_valid.shape[0], time_steps, num_features))

    # x_test = x_test.reshape((x_test.shape[0], time_steps, num_features))

    return x_train, y_train, x_test, y_test


def data_input():

    scaled, scalar = scale_data2()  # using scalar to inv_transform later
    whole = series_to_supervised(values=scaled, n_in=7, n_out=1)

    x_train, y_train, x_test, y_test = partition(whole)

    print("train_X shape:", x_train.shape)
    print("train_y shape:", y_train.shape)
    print("test_X shape:", x_test.shape)
    print("test_y shape:", y_test.shape)

    return x_train, y_train, x_test, y_test, scalar


# Linear Regression model define
def linear():
    x_train, y_train, x_test, y_test, scalar = data_input()
    # y_test = y_test.reshape((len(y_test), 1))

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model,scalar,  x_test, y_test


def Model_prediction(features=10, time_steps=7):

    model, scalar, x_test, y_test = linear()

    x_test = x_test.reshape(x_test.shape[0], features * time_steps)
    predictions = (model.predict(x_test)).reshape(len(y_test), 1)

    inv_predictions = np.concatenate((predictions, x_test[:, (1 - features):]), axis=1)
    inv_whole_test = scalar.inverse_transform(inv_predictions)
    inv_predictions = inv_whole_test[:, 0]

    y_test = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((y_test, x_test[:, (1 - features):]), axis=1)
    inv_y = scalar.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # for i, prediction in enumerate(predictions):
    #     print('Predicted: %s, Target: %s' % (inv_predictions[i], inv_y[i]))
    avg = np.average(inv_y)
    rmse = sqrt(mean_squared_error(inv_predictions, inv_y))
    error_percentage = rmse / avg
    print("Test Root Mean Square Error of Linear Regression: %.3f" % rmse)
    print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
    plt.figure(1)
    plt.plot(inv_y, label='True Label')
    plt.plot(inv_predictions, label='Prediction Value')
    plt.legend(loc='best')
    plt.xlabel('Days')
    plt.ylabel('Consumption')
    plt.title('Single Step Linear Regression')
    plt.savefig('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\Train_result\\Linear_Regression.png')
    plt.show()


Model_prediction()
