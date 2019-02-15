'''
This function divide the dataset into train//valid//test set in a fixed numbers
'''


def partition(data): # used to divide the train and test data
    values = data.values
    num_trained = 600
    num_valid = 121

    train_set = values[0: num_trained, :]
    valid_set = values[num_trained:num_trained+num_valid, :]
    test_set = values[num_trained+num_valid:, :]

    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]

    x_valid = valid_set[:, :-1]
    y_valid = valid_set[:, -1]

    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    # y_test_final = test_set[:, -1]

    # reshape to teh [samples, time_steps, num_of_features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

    # y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))

    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # y_test  = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))

    return x_train, y_train, x_valid, y_valid, x_test, y_test
