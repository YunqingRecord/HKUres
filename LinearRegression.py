from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Transform_to_Training import scale_data, read_file, get_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from numpy import concatenate
from math import sqrt


'''
In this file, I use the LinearRegression Model to fit the data, predicting the next_day consumption,
by given the last day parameters.
 
'''


# Linear Regression model define
def linear():
    x_train, y_train, x_valid, y_valid, x_test, y_test, scalar = get_data()
    y_test = y_test.reshape((len(y_test), 1))
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = (model.predict(x_test)).reshape(len(y_test), 1)
    inv_predictions = concatenate((x_test, predictions), axis=1)
    inv_whole_test = scalar.inverse_transform(inv_predictions)
    inv_predictions = inv_whole_test[:, -1]
    inv_y = concatenate((x_test, y_test), axis=1)
    inv_y = scalar.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # for i, prediction in enumerate(predictions):
    #     print('Predicted: %s, Target: %s' % (inv_predictions[i], inv_y[i]))
    print('R2 square of LinearRegression:', r2_score(inv_y, inv_whole_test[:, -1]))
    rmse = sqrt(mean_squared_error(inv_predictions, inv_y))
    print('RMSE of Linear Regression : ', rmse)
    plt.figure(1)
    plt.plot(inv_predictions, label='Prediction Value')
    plt.plot(inv_y, label='True Label')
    plt.legend(loc='best')
    plt.xlabel('Days')
    plt.ylabel('Consumption')
    plt.title('Single Step Linear Regression')
    plt.savefig('C:\\Users\\Yunqing\\Desktop\\dissertation of HKU\\HKUresdata\\Train_result\\Linear_Regression.png')
    plt.show()


linear()
