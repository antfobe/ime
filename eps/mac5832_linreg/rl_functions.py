import numpy as np
from util import randomize_in_place

def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)

def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    # YOUR CODE HERE:
    X_out = (X - np.mean(X))/np.std(X)
    #raise NotImplementedError
    # END YOUR CODE

    return X_out

def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """

    # YOUR CODE HERE:
    J = (np.matmul(
            (np.dot(X,w) - y).T, 
            np.dot(X,w) - y) / y.size)[0][0]
    #raise NotImplementedError
    # END YOUR CODE

    return J

def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    # YOUR CODE HERE:
    y_estimate = X.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    grad = -(1.0/len(X)) * error.dot(X) * 2
    grad = grad.reshape(grad.size, 1)
    #raise NotImplementedError
    # END YOUR CODE
    return grad

def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]

    # YOUR CODE HERE:
    for i in range(0, num_iters): 
        w = w - learning_rate * compute_wgrad(X, y, w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))
    #raise NotImplementedError
    # END YOUR CODE

    return w, weights_history, cost_history

def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    # YOUR CODE HERE:
    m = np.random.choice(y.size, batch_size)
    X = X[m,:]
    y = y[m,:]
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]

    for i in range(0, num_iters): 
        w = w - learning_rate * compute_wgrad(X, y, w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))
    #raise NotImplementedError
    # END YOUR CODE

    return w, weights_history, cost_history
