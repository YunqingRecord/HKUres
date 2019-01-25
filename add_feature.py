import csv
from pandas import DataFrame
import os
from pandas import read_csv
from pandas import concat

'''
Firstly, add feature,
then, add label and 
lastly make it normalized
'''


def add_feature(filename1, filename2):
    '''
    :param filename: filename1: the original data sampled by bluesky
                    filename2: the sampled features from the  HKO
    :return:  the new-feature added csv files
    '''
    '''
    the Origin File Value
    '''
    dataset = read_csv(filename1, index_col=0,
                       usecols=['time', 'total', 'ac', 'light', 'socket', 'other', 'mixed_usage',
                                'next_holiday', 'temperature_max', 'temperature_min'])
    values = dataset.values
    values = values.astype('float32')
    values = DataFrame(values)
    '''
    the new_feature Value 2
    '''
    name = ['total', 'ac', 'light', 'socket', 'other', 'mixed_usage', 'next_holiday', 'temperature_max',
            'temperature_min', 'pressure', 'dew_temp', 'humidity', 'cloudiness', 'rainfall']
    new_data = read_csv(filename2, index_col=0,
                         usecols=['date', 'pressure', 'dew_temp', 'humidity', 'cloudiness', 'rainfall'])
    values2 = new_data.values
    values2 = DataFrame(values2)
    final = [values, values2]
    combination = concat(final, axis=1)
    combination.columns = name
    return combination


def read_file2(first_path = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\floors_2018_06_18\\'):
    # read all files iteratively and process into new csv file
    csv_list = os.listdir(first_path)
    i = 0
    for csv_file in csv_list:
        i += 1
        filename1 = first_path + csv_file
        filename2 = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\weather_record.csv'
        values = DataFrame(add_feature(filename1, filename2))
        values.to_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\add_feature\\'+str(csv_file))


read_file2()