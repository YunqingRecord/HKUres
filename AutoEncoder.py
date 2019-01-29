from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv

'''
construct the dataset // train and test
'''

filename = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\add_feature\\11_F.csv'
data = DataFrame(read_csv(filename, index_col=0))
data = data.values
X_train = (data[:700])
X_test = data[700:]
X_train = X_train.reshape(X_train.shape[0], 1,  X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

encoding_dim = 10


model = Sequential()
model.add(Dense(encoding_dim, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(12))

model.compile(loss='mae', optimizer='adam')

history = model.fit(X_train, X_train, epochs= 100, batch_size= 16, validation_data= (X_test, X_test))

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
plt.show()

'''
Finish of training
'''