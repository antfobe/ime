#!/usr/bin/python3

import unittest
from unittest.util import safe_repr
import numpy as np
import time
from util import r_squared, get_housing_prices_data, grad_check
from util import add_feature_ones, randomize_in_place
from rl_functions import standardize, compute_cost, compute_wgrad
from rl_functions import batch_gradient_descent, stochastic_gradient_descent
from rl_functions import linear_regression_prediction


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.
    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestEP2(unittest.TestCase):
    """
    Class that test the functions from rl_functions module
    """
    @classmethod
    def setUpClass(cls):
        X, y = get_housing_prices_data(N=350)
        randomize_in_place(X, y)
        cls.train_X = X[0:250]
        cls.train_y = y[0:250]
        cls.valid_X = X[250:300]
        cls.valid_y = y[250:300]
        cls.test_X = X[300:]
        cls.test_y = y[300:]
        cls.total_score = 0

    def assertTrue(self, expr, msg=None, score=0):
        """Check that the expression is true."""
        if not expr:
            msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr)) # noqa
            raise self.failureException(msg)
        else:
            TestEP2.total_score += score

    def test_exercise_1(self):
        try:
            toy_X = np.array([[1100.3, 2.4, 34.34],
                              [2300.3, 1.4, 442.23]])
            toy_y = np.array([[1000.2], [2000.5]])
            toy_X_norm = standardize(toy_X)
            toy_y_norm = standardize(toy_y)
            xmean, xstd = np.mean(toy_X_norm), np.std(toy_X_norm)
            ymean, ystd = np.mean(toy_y_norm), np.std(toy_y_norm)
            self.assertTrue(-1 <= xmean < 0, score=1 / 4)
            self.assertTrue(0 <= ymean < 1, score=1 / 4)
            self.assertTrue(0.9 <= xstd <= 1, score=1 / 4)
            self.assertTrue(0.9 <= ystd <= 1, score=1 / 4)
        except NotImplementedError:
            self.fail('Exercício 1) Falta fazer!')

    def test_exercise_2(self):
        try:
            toy_w = np.array([[1], [1], [2]])
            toy_X = np.array([[2, 3, 1],
                              [5, 1, 2]])
            toy_y = np.array([[1], [1]])
            self.assertTrue(compute_cost(toy_X, toy_y, toy_w) == 58.5,
                            score=1)
        except NotImplementedError:
            self.fail('Exercício 2) Falta fazer!')

    def test_exercise_3(self):
        try:
            toy_w1 = np.array([[1.], [2.], [1.], [2.]])
            toy_X1 = np.array([[2., 3., 1., 2.],
                              [5., 1., 1., 2.]])
            toy_y1 = np.array([[1.], [-1.]])
            toy_w2 = np.array([[-100.22], [20002.1], [102.5]])
            toy_X2 = np.array([[2111.3, -2223., 404.0],
                              [5222., -22221., 3.3]])
            toy_y2 = np.array([[122.], [221.]])
            toy_w3 = np.array([[-10.22], [-3.1]])
            toy_X3 = np.array([[1.3, -1.2],
                              [2.2, -2.1],
                              [-2.3, -5.5],
                              [3.2, 8.1],
                              [3.3, -1.1],
                              [-3.4, -2.22],
                              [2.23, -4.4],
                              [5.2, -2.3]])
            toy_y3 = np.array([[10.3],
                               [23.3],
                               [10.1],
                               [-20.2],
                               [-10.2],
                               [20.2],
                               [-14.4],
                               [-30.3]])
            self.assertTrue(grad_check(toy_X1,
                                       toy_y1,
                                       toy_w1,
                                       compute_cost,
                                       compute_wgrad), score=1 / 3)
            self.assertTrue(grad_check(toy_X2,
                                       toy_y2,
                                       toy_w2,
                                       compute_cost,
                                       compute_wgrad), score=1 / 3)
            self.assertTrue(grad_check(toy_X3,
                                       toy_y3,
                                       toy_w3,
                                       compute_cost,
                                       compute_wgrad), score=1 / 3)
        except NotImplementedError:
            self.fail('Exercício 3) Falta fazer!')

    def test_exercise_4(self):
        try:
            initial_w = np.array([[15], [-35.3]])
            train_X_norm = standardize(self.train_X)
            train_y_norm = standardize(self.train_y)
            train_X_1 = add_feature_ones(train_X_norm)
            learning_rate = 0.8
            iterations = 20000
            init = time.time()
            w, weights_history, cost_history = batch_gradient_descent(train_X_1, # noqa
                                                                      train_y_norm, # noqa
                                                                      initial_w, # noqa
                                                                      learning_rate, # noqa
                                                                      iterations) # noqa
            init = time.time() - init
            self.assertTrue(cost_history[-1] < cost_history[0], score=1 / 5)
            self.assertTrue(type(w) == np.ndarray, score=1 / 5)
            self.assertTrue(len(weights_history) == len(cost_history), score=1 / 5) # noqa
            self.assertTrue(init < 2.5, score=1 / 5)
            test_X_norm = standardize(self.test_X)
            test_X_1 = add_feature_ones(test_X_norm)
            prediction = linear_regression_prediction(test_X_1, w)
            prediction = (prediction * np.std(self.train_y)) + np.mean(self.train_y) # noqa
            r_2 = r_squared(self.test_y, prediction)
            self.assertTrue(0.3 < r_2, score=1 / 5)
        except NotImplementedError:
            self.fail('Exercício 4) Falta fazer!')

    def test_exercise_5(self):
        try:
            initial_w = np.array([[15], [-35.3]])
            train_X_norm = standardize(self.train_X)
            train_y_norm = standardize(self.train_y)
            train_X_1 = add_feature_ones(train_X_norm)
            learning_rate = 0.8
            iterations = 2000
            batch_size = 36
            init = time.time()
            w, weights_history, cost_history = stochastic_gradient_descent(train_X_1, # noqa
                                                                           train_y_norm, # noqa
                                                                           initial_w, # noqa
                                                                           learning_rate, # noqa
                                                                           iterations, # noqa
                                                                           batch_size) # noqa
            init = time.time() - init
            self.assertTrue(cost_history[-1] < cost_history[0], score=1 / 5)
            self.assertTrue(type(w) == np.ndarray, score=1 / 5)
            self.assertTrue(len(weights_history) == len(cost_history), score=1 / 5) # noqa
            self.assertTrue(init < 2.5, score=1 / 5)
            test_X_norm = standardize(self.test_X)
            test_X_1 = add_feature_ones(test_X_norm)
            prediction = linear_regression_prediction(test_X_1, w)
            prediction = (prediction * np.std(self.train_y)) + np.mean(self.train_y) # noqa
            r_2 = r_squared(self.test_y, prediction)
            self.assertTrue(0.3 < r_2, score=1 / 5)
        except NotImplementedError:
            self.fail('Exercício 5) Falta fazer!')


if __name__ == '__main__':
    run_test(TestEP2)
    time.sleep(0.1)
    total_score = (TestEP2.total_score / 5) * 10
    print("\nEP2 total_score = ({:.1f} / 10.0)".format(total_score))
