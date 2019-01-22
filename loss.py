import tensorflow as tf
import keras.backend as K
import numpy as np
# import matplotlib.pyplot as plt


def loss(y_true, y_pred):   # designed loss function
    err = y_true - y_pred
    lost = 0

    less0 = tf.less_equal(err, 0)  # find elements less than 0 as True
    cast1 = tf.cast(less0, dtype=tf.float32)  # convert bool to 0/1

    greater0 = tf.greater(err, 0)  # find elements greater than 0 as True
    cast2 = tf.cast(greater0, dtype=tf.float32) # convert bool to 0/1

    err1 = tf.where(less0, err, cast1)  # elements less than 0
    err2 = tf.where(greater0, err, cast2)  # elements greater than 0

    lost += 1 - K.exp((-K.log(0.5)) * (err1 / 20))
    lost += 1 - K.exp((K.log(0.5)) * (err2 / 5))

    return K.mean(lost)


def score(err):   # score function , plot the scatters
    # err = y_true - y_pred
    if err < 0:
        score = np.exp(-np.log(0.5)*(err/20))
    elif err > 0:
        score = np.exp(np.log(0.5)*(err/5))
    else:
        score = 1
    return score



