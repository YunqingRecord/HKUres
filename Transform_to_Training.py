from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    scaled = scalar.fit_transform(values)
    df = DataFrame(scaled, columns=['total', 'ac', 'light', 'socket', 'temperature_max',
                                    'temperature_min', 'pressure', 'dew_temp', 'next_consumption'])
    df.to_csv()
    print(df.head())



