from scipy.stats import pearsonr
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os

total = []
ac = []
light = []
socket = []
next_holiday = []
temperature_max = []
temperature_min = []
pressure = []
dew_temp = []
humidity = []
cloudiness = []
rainfall = []

column_name = ['total', 'ac', 'light', 'socket', 'next_holiday',
               'temperature_max', 'temperature_min', 'pressure',
               'dew_temp', 'humidity', 'cloudiness', 'rainfall', 'next_consumption']


def pearson(filepath='C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\with_label\\'):

    csv_list = os.listdir(filepath)
    pear = []
    for csv_file in csv_list:
        dataset = read_csv(filepath+csv_file, index_col=0)
        for i in range(0, 12):
            x = dataset[column_name[i]]
            y = dataset[column_name[-1]]
            m = pearsonr(x, y)
            pear.append(m)
    return pear


def output_pear():
    '''
    :param : distribute the pearson ana answer into list resp. and plot them to visualize the answer
    :return: the visualized answer
    '''
    pear = pearson()
    ans = [total, ac, light, socket, next_holiday, temperature_max,
           temperature_min, pressure, dew_temp, humidity, cloudiness, rainfall]
    for i in range(0, 12):
        j = 0
        j += i
        while j < 360:
            ans[i].append(pear[j])
            j += 12
    return ans


ans = output_pear()
print(len(ans))
for i in range(12):
    plt.figure(i)
    temp = []
    for j in range(30):
        temp.append(ans[i][j][0])
    x = np.linspace(1, 30, 30)
    plt.xlim(1, 30)
    plt.ylim(-1, 1)
    plt.scatter(x, temp)
    plt.title('Pearson_corr of '+column_name[i])
    plt.savefig(
        'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\Pearson_result\\' + column_name[i] + '.png')
    plt.show()
