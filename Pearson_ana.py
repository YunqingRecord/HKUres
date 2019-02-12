from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame


'''
read files to be analyzed using pearson:
'''

dataset =  read_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\with_label\\11_F.csv')

dataset2 = DataFrame(dataset).values
column_name = list(dataset)

pear = []
m, n = dataset.shape
for i in range(1, 13):
    x = dataset2[:][i]
    y = dataset2[:][n-1]
    m = pearsonr(x, y)
    if m[0] == 'nan':
        pear.append(('nan', 'nan'))
    pear.append(m)
    print(column_name[i], m)
