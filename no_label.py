from pandas import read_csv
import numpy as np
import os
from pandas import DataFrame


def read_file(first_path='C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\with_label\\'):
    # read all added features files iteratively and process into new csv file
    csv_list = os.listdir(first_path)
    for csv_file in csv_list:
        filename = first_path + csv_file
        values = DataFrame(read_csv(filename,
                           usecols=['total', 'ac', 'light', 'socket', 'next_holiday',
                                    'temperature_max', 'temperature_min', 'pressure',
                                    'dew_temp', 'humidity', 'cloudiness', 'rainfall']))
        values.to_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\no_label\\'+'n'+str(csv_file))

