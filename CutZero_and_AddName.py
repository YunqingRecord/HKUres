from pandas import DataFrame
from pandas import read_csv
import pandas
import numpy as np
import os


'''
Use this function to cut ZeroValue in Floor Function csv,

and recover the origin name to the csv file
'''


def read_file3(first_path='C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\add_feature\\'):
    # read all files iteratively and process into new csv file
    csv_list = os.listdir(first_path)
    i = 0
    for csv_file in csv_list:
        i += 1
        filename = first_path + csv_file
        values = DataFrame(cut_zero(filename))
        values.columns = [['total', 'ac', 'light', 'socket', 'next_holiday', 'temperature_max',
                           'temperature_min', 'pressure', 'dew_temp', 'humidity', 'cloudiness', 'rainfall']]
        values.to_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\Floor_revised\\'+str(csv_file))


def cut_zero(filename):

    '''
    :param filename: files in loop to be cut
    :return: the recovered file
    '''

    dataset = DataFrame(read_csv(filename,
                        usecols=['total', 'ac', 'light', 'socket', 'next_holiday', 'temperature_max',
                                 'temperature_min', 'pressure', 'dew_temp', 'humidity', 'cloudiness', 'rainfall']))
    return dataset


read_file3()