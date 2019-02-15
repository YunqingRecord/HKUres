from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from Partition_dataset import partition
import os


def read_file(filepath = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\One_step_origin_with_label\\'):
    # read all added features files iteratively and process into new csv file
    csv_list = os.listdir(filepath)
    i = 0
    for csv_file in csv_list:
        i += 1
        filename = filepath + csv_file
        dataset = read_csv(filename, usecols=['total', 'ac', 'light', 'socket', 'temperature_max',
                                              'temperature_min', 'pressure', 'dew_temp', 'next_consumption'])
        return dataset


def scale_data():
    data = read_file()
    values = data.values
    scalar = StandardScaler()
    # scalar = MinMaxScaler(feature_range=(1, 2))
    scaled = scalar.fit_transform(values)
    df = DataFrame(scaled, columns=['total', 'ac', 'light', 'socket', 'temperature_max',
                                    'temperature_min', 'pressure', 'dew_temp', 'next_consumption'])
    return df, scalar


def get_data():
    df, scalar = scale_data()
    x_train, y_train, x_valid, y_valid, x_test, y_test = partition(df)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scalar

