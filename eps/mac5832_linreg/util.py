import numpy as np


def grad_check(X, y, w, compute_cost, compute_wgrad, h=1e-4, verbose=False):
    """
    Check gradients for linear regression.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param h: small variation
    :type h: float
    :return: gradient test
    :rtype: boolean
    """
    Jw = compute_cost(X, y, w)
    grad = compute_wgrad(X, y, w)
    passing = True
    d = w.shape[0]
    for i in range(d):
        w_plus_h = np.array(w, copy=True)
        w_plus_h[i] = w_plus_h[i] + h
        Jw_plus_h = compute_cost(X, y, w_plus_h)
        w_minus_h = np.array(w, copy=True)
        w_minus_h[i] = w_minus_h[i] - h
        Jw_minus_h = compute_cost(X, y, w_minus_h)
        numgrad_i = (Jw_plus_h - Jw_minus_h) / (2 * h)
        reldiff = abs(numgrad_i - grad[i]) / max(1, abs(numgrad_i), abs(grad[i])) # noqa
        if reldiff > 1e-5:
            passing = False
            if verbose:
                msg = """
                Seu gradiente = {0}
                Gradiente numérico = {1}""".format(grad[i], numgrad_i)
                print("            " + str(i) + ": " + msg)
                print("            Jw = {}".format(Jw))
                print("            Jw_plus_h = {}".format(Jw_plus_h))
                print("            Jw_minus_h = {}\n".format(Jw_minus_h))

    if passing and verbose:
        print("Gradiente passando!")

    return passing


def add_feature_ones(X):
    """
    Returns the ndarray 'X' with the extra
    feature column containing only 1s.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: output array
    :rtype: np.ndarray(shape=(N, d+1))
    """
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def get_housing_prices_data(N):
    """
    Generates artificial linear data,
    where x = square meter, y = house price

    :param N: data set size
    :type N: int
    :param verbose: param to control print
    :type verbose: bool
    :return: design matrix, regression targets
    :rtype: np.array, np.array
    """
    cond = False
    while not cond:
        x = np.linspace(90, 1200, N)
        gamma = np.random.normal(30, 10, x.size)
        y = 50 * x + gamma * 400
        x = x.astype("float32")
        x = x.reshape((x.shape[0], 1))
        y = y.astype("float32")
        y = y.reshape((y.shape[0], 1))
        cond = min(y) > 0
    return x, y


def r_squared(y, y_hat):
    """
    Calculate the R^2 value

    :param y: regression targets
    :type y: np array
    :param y_hat: prediction
    :type y_hat: np array
    :return: r^2 value
    :rtype: float
    """
    y_mean = np.mean(y)
    ssres = np.sum(np.square(y - y_mean))
    ssexp = np.sum(np.square(y_hat - y_mean))
    sstot = ssres + ssexp
    return 1 - (ssexp / sstot)


def randomize_in_place(list1, list2, init=0):
    """
    Function to randomize two lists in the same way.

    :param list1: list
    :type list1: list or np.array
    :param list2: list
    :type list2: list or np.array
    :param init: seed
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)