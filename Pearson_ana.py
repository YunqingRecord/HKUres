from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame


'''
read files to be analyzed using pearson:

'''

dataset =  read_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\addlabel\\1.csv', index_col=0)

dataset = DataFrame(dataset)
pear = []
m, n = dataset.shape
for i in range(9):
    x = dataset[str(i)]
    y = dataset[str(n-1)]
    m = pearsonr(x, y)
    if m[0] == 'nan':
        pear.append(('nan', 'nan'))
    pear.append(m)
    print(m)

