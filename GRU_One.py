import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
import os
from loss import loss


def read_file(filepath = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\One_step_origin_with_label\\'):
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


def scale_data():
    data = read_file()
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
    valid_set = values[num_trained:num_trained+num_valid, :]
    test_set = values[num_trained+num_valid-26:, :]

    x_train = train_set[:, :time_steps*num_features]
    y_train = train_set[:, -num_features]

    # x_valid = valid_set[:, :time_steps*num_features]
    # y_valid = valid_set[:, time_steps*num_features]

    x_test = test_set[:, :time_steps*num_features]
    y_test = test_set[:, -num_features]

    # reshape to teh [samples, time_steps, num_of_features]
    x_train = x_train.reshape((x_train.shape[0], time_steps, num_features))
    # # y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    # x_valid = x_valid.reshape((x_valid.shape[0], time_steps, num_features))

    x_test = x_test.reshape((x_test.shape[0], time_steps, num_features))

    return x_train, y_train, x_test, y_test


def data_input():

    scaled, scalar = scale_data()  # using scalar to inv_transform later
    whole = series_to_supervised(values=scaled, n_in=7, n_out=1)

    x_train, y_train, x_test, y_test = partition(whole)

    print("train_X shape:", x_train.shape)
    print("train_y shape:", y_train.shape)
    print("test_X shape:", x_test.shape)
    print("test_y shape:", y_test.shape)

    return x_train, y_train, x_test, y_test, scalar


def Model():
    x_train, y_train, x_test, y_test, scalar = data_input()
    model = Sequential()
    model.add(GRU(30, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True, return_sequences=True)) #, stateful=True))
    model.add(GRU(30))  # , stateful=True))
    model.add(Dense(20))
    model.add(Dense(12))
    model.add(Dense(5))
    model.add(Dense(1))
    sgd = SGD(lr=0.0001, decay=1e-7, momentum=0.95, nesterov=True)
    model.compile(optimizer='Adam', loss=loss)
    history = model.fit(x_train, y_train, epochs=3000, batch_size=1, shuffle=False)
    plt.figure(1)
    plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train_loss"], loc="upper right")

    # plt.figure(2)
    # plt.plot(history.history['acc'], label='train_acc')
    # # plt.plot(history.history['val_acc'], label='val_acc')

    # plt.title("Model Accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("epoch")
    # plt.legend(["train_acc"], loc="upper right")


    plt.show()
    return model, scalar, x_test, y_test


def Model_Prediction(features=10, time_steps=7):

    model, scalar, x_test, y_test = Model()
    y_pred = model.predict(x_test, batch_size=1)
    x_test = x_test.reshape(x_test.shape[0], features*time_steps)

    '''
    inverse scale using scalar saved before
    '''
    inv_y_pred = np.concatenate((y_pred, x_test[:, (1 - features):]), axis=1)
    inv_y_pred = scalar.inverse_transform(inv_y_pred)
    inv_y_pred = inv_y_pred[:, 0]
    # invert scaling for actual
    test_y = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((test_y, x_test[:, (1 - features):]), axis=1)
    inv_y = scalar.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_y_pred))
    avg = np.average(inv_y)
    error_percentage = rmse / avg

    print("Test Root Mean Square Error: %.3f" % rmse)
    print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))

    # calculate average error percentage
    avg = np.average(inv_y)
    error_percentage = rmse / avg

    print("Test Root Mean Square Error: %.3f" % rmse)
    print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
    plt.figure(2)
    plt.plot(inv_y, label="Actual Consumption")
    plt.plot(inv_y_pred, label="Predicted Consumption")
    plt.xlabel('Days')
    plt.ylabel('Consumption')
    plt.title('Single Step GRU')
    plt.legend(loc='best')
    plt.show()

    return inv_y, inv_y_pred, rmse, error_percentage


Model_Prediction()

