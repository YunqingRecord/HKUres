import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.layers as layers
import keras.models
from keras.layers.convolutional import Conv1D
from keras.utils.vis_utils import plot_model
from keras.optimizers import adam, adadelta, adagrad, SGD

from math import sqrt
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from pandas import concat
from sklearn.metrics import mean_squared_error
import os
from loss import loss


# loss function: final target
def central_high(y_true, y_pred):
    err = (tf.argmax(y_true, axis=-1) - tf.argmax(y_pred, axis=-1))*5
    err = tf.cast(tf.reshape(err, (-1,2000,1)),dtype=tf.float32)
    loss = 0

    less0 = tf.less_equal(err, 0) # find elements less than 0 as True
    cast1 = tf.cast(less0, dtype=tf.float32)  # convert bool to 0/1

    greater0 = tf.greater(err, 0) # find elements greater than 0 as True
    cast2 = tf.cast(greater0, dtype=tf.float32) # convert bool to 0/1

    err1 = tf.where(less0, err, cast1) # elements less than 0
    err2 = tf.where(greater0, err, cast2) # elements greater than 0

    loss += 1 - K.exp((-K.log(0.5)) * (err1 / 5))
    loss += 1 - K.exp((K.log(0.5)) * (err2 / 20))

    loss = K.mean(loss)
    return loss


def score(y_true, y_pred):
    err = (tf.argmax(y_true, axis=-1) - tf.argmax(y_pred, axis=-1)) * 5
    err = tf.cast(tf.reshape(err, (-1, 2000, 1)), dtype=tf.float32)
    score = 0

    less0 = tf.less_equal(err, 0)  # find elements less than 0 as True
    cast1 = tf.cast(less0, dtype=tf.float32)  # convert bool to 0/1

    greater0 = tf.greater(err, 0)  # find elements greater than 0 as True
    cast2 = tf.cast(greater0, dtype=tf.float32)  # convert bool to 0/1

    err1 = tf.where(less0, err, cast1)  # elements less than 0
    err2 = tf.where(greater0, err, cast2)  # elements greater than 0

    score += K.exp((-K.log(0.5)) * (err1 / 5))
    score += K.exp((K.log(0.5)) * (err2 / 20))

    score = K.mean(score)
    return score


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
    dataset = DataFrame(read_csv(filename, usecols=['total', 'ac', 'light', 'socket', 'next_holiday', 'temperature_max',
                                          'temperature_min', 'pressure', 'dew_temp', 'cold_season']))
    # dataset['ac'] = dataset['ac']/1000
    # dataset['socket'] = dataset['socket']/1000
    # dataset['light'] = dataset['light']/1000

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
    x_train = x_train.reshape((x_train.shape[0], 1, 70))
    # # y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    # x_valid = x_valid.reshape((x_valid.shape[0], time_steps, num_features))

    x_test = x_test.reshape((x_test.shape[0], 1, 70))

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


def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    x = Conv1D(filters1, kernel_size)(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('tanh')(x)

    x = Conv1D(filters2, kernel_size, padding='same')(x)
    # x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('tanh')(x)

    x = Conv1D(filters3, kernel_size)(x)
    # x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('tanh')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=2):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    x = Conv1D(filters1, kernel_size, strides=strides)(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis)(x)
    # x = layers.Activation('tanh')(x)

    x = Conv1D(filters2, kernel_size, padding='same')(x)
    # x = layers.BatchNormalization(axis=bn_axis)(x)
    # x = layers.Activation('tanh')(x)

    x = Conv1D(filters3, kernel_size)(x)
    # x = layers.BatchNormalization(axis=bn_axis)(x)

    shortcut = Conv1D(filters3, kernel_size, strides=strides)(input_tensor)
    # shortcut = layers.BatchNormalization(axis=bn_axis)(shortcut)

    x = layers.add([x, shortcut])
    # x = layers.Activation('tanh')(x)
    return x


def ResNet1D(input_tensor=None, strides=1):

    x = Conv1D(input_shape=(1, 70), filters=128, kernel_size=1, strides=strides, padding='valid')(input_tensor)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('tanh')(x)
    x = conv_block(input_tensor=x, kernel_size=1, filters=[128, 128, 64], strides=1)
    # x = identity_block(x, 1, [4, 4, 8], stage=2, block='b')
    x = identity_block(x, 1, [64, 32, 64])

    x = conv_block(x, 1, [8, 8, 32])
    x = identity_block(x, 1, [8, 8, 32])
    x = identity_block(x, 1, [8, 8, 32])
    x = identity_block(x, 1, [8, 8, 32])
    #
    x = conv_block(x, 1, [16, 16, 32])
    x = identity_block(x, 1, [16, 16, 32])
    x = identity_block(x, 1, [16, 16, 32])
    x = identity_block(x, 1, [16, 16, 32])
    x = identity_block(x, 1, [16, 16, 32])
    # x = identity_block(x, 1, [128, 128, 512], stage=4, block='c')
    # x = identity_block(x, 1, [128, 128, 512], stage=4, block='d')
    # x = identity_block(x, 1, [128, 128, 512], stage=4, block='e')
    # x = identity_block(x, 1, [128, 128, 512], stage=4, block='f')

    # x = conv_block(x, 1, [256, 256, 1024], stage=5, block='a')
    # x = identity_block(x, 1, [256, 256, 1024], stage=5, block='b')
    # x = identity_block(x, 1, [256, 256, 1024], stage=5, block='c')
    x = conv_block(x, 1, [32, 32, 64])
    x = identity_block(x, 1, [32, 32, 64])
    x = identity_block(x, 1, [32, 32, 64])

    x = layers.Flatten()(x)

    return x


def Model():
    x_train, y_train, x_test, y_test, scalar = data_input()

    I1 = layers.Input(shape=(1, 70))

    x_spec_res = (ResNet1D(input_tensor=I1, strides=1))
    y1 = layers.Dense(units=12)(x_spec_res)
    y1 = layers.Dense(units=8)(y1)
    y_hat = layers.Dense(units=1)(y1)

    model = keras.models.Model(inputs=I1, outputs=y_hat)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    history=model.fit(x_train, y_train, batch_size=16, epochs=200)
    # plot_model(model, to_file='r.png')

    # plt.plot(history.history['val_loss'], label='val_loss')

    plt.figure(1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train_loss"], loc="upper right")

    plt.show()
    return model, scalar, x_test, y_test


def Model_Prediction(features=10, time_steps=7):

    model, scalar, x_test, y_test = Model()
    y_pred = model.predict(x_test, batch_size=16)
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
    plt.title('One_dim Resnet')
    plt.legend(loc='best')
    plt.show()

    return inv_y, inv_y_pred, rmse, error_percentage


Model_Prediction()