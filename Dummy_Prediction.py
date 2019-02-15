from sklearn.linear_model import LinearRegression
from Transform_to_Training import get_data
import matplotlib.pyplot as plt
import numpy as np
from  pandas import read_csv
import os
from math import sqrt
from sklearn.metrics import mean_squared_error

'''
In this file, I simply use the Dummy prediction to output the data as the next_day electricity consumption,
by not given the last day parameters. This is the control group

'''


def read_file(filepath = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\One_step_origin_with_label\\'):
    # read all added features files iteratively and process into new csv file
    csv_list = os.listdir(filepath)
    i = 0
    for csv_file in csv_list:
        i += 1
        filename = filepath + csv_file
        dataset = read_csv(filename, usecols=['total', 'next_consumption'])
        dataset = dataset.values
        return dataset


def dummy():
    '''
    :parameter: using the last day consumption value as the output to be the control group
    :return: accuracy
    '''
    data = read_file()
    x_test = data[722:, 0]
    y_test = data[722:, 1]
    rmse = sqrt(mean_squared_error(x_test, y_test))
    print('RMSE of Dummy Prediction: ', rmse)
    plt.figure(1)
    plt.plot(x_test, label='Dummy Value')
    plt.plot(y_test, label='True Label')
    plt.title('Dummy Prediction')
    plt.legend(loc='best')
    plt.xlabel('Days')
    plt.ylabel('Consumption')
    # plt.title('Single Step Linear Regression Accuracy: 92.44%')
    plt.savefig('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\Train_result\\Dummy.png')
    plt.show()


dummy()