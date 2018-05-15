#NUSP7630573

import numpy as np

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new row with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :return: prediction
    :rtype: np.ndarray(shape=(N, 1))
    """
    # YOUR CODE HERE:
    # PS: It's ... it's beautiful! (seriously tho, to hell with oo)
    try:
        prediction = np.matmul(
                    X,
                    np.matmul(
                        np.matmul(
                            np.linalg.inv(
                                np.matmul(
                                    X.transpose(),
                                    X)), 
                            X.transpose()), 
                        y)
        );
    except ValueError:
        print 'Value Error - is input data a correctly shaped ndarray?'

    # END YOUR CODE
    return prediction
