from keras.layers import Input
from keras.models import Model
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv

'''
construct the dataset // train and test
'''

filename = 'C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\no_label\\n11_F.csv'
data = read_csv(filename, index_col=0)
data = data.values
X_train = (data[:700])
X_test = data[700:]
X_train = X_train.reshape(X_train.shape[0], 1,  X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

encoding_dim = 10


model = Sequential()
model.add(Dense(encoding_dim, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(12))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='mae', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, X_train, epochs=100, batch_size=128, validation_data= (X_test, X_test))
model.summary()
prediction = model.predict(data)
(DataFrame(prediction)).to_csv('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\decoded\\11_re.csv')

pyplot.figure(1)
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"], loc="upper right")

pyplot.figure(2)
pyplot.plot(history.history['acc'], label='train_acc')
pyplot.plot(history.history['val_acc'], label='val_acc')

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc", "val_acc"], loc="upper right")

plt.show()

'''
Finish of training
'''