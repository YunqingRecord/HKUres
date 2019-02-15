from pandas import DataFrame
import os
from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\with_label\\11_F.csv', index_col=0)
value = dataset.values
# specify columns to plot
group1 = [0, 1, 2, 3, 4, 5, -1]
group2 = [6, 7, 8, 9, 10, 11, -1]
i = 1
j = 1
# plot each column
pyplot.figure(1)
for group in group1:
    pyplot.subplot(len(group1), 1, i)
    pyplot.plot(value[:365, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1

pyplot.figure(2)
for group in group2:
    pyplot.subplot(len(group2), 1, j)
    pyplot.plot(value[:365, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    j += 1
pyplot.show()