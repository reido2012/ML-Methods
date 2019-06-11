import numpy as np
import time


class LinearRegression:

    def __init__(self, x_train, y_train, x_test, y_test, step_size=0.0001, num_iterations=40000):
        self.convergence_threshold = 1*10**-10
        self.decreasing = False
        self.converged = False
        self.step_size = step_size
        self.original_x_train_shape = x_train.shape
        self.x_train = self.process_inputs(x_train)
        self.x_test = self.process_inputs(x_test)
        self.weights = np.ones(x_train.shape[1] + 1)
        self.y_train = y_train
        self.y_test = y_test
        self.num_iterations = num_iterations
        self.optimal_hyperparams = {'step_size': None, 'num_iterations': None}
        self.eval_values = {'RMSE': None, 'R2': None, 'Time': None}
        self.all_hyperparam_tests = []

    def reset_init_variables(self):
        self.decreasing = False
        self.converged = False
        self.weights = np.ones(self.original_x_train_shape[1] + 1)

    def _least_squares_loss(self, x_train, y_train, weights):
        return np.sum((x_train.dot(weights) - y_train) ** 2) / (2 * x_train.shape[0])

    def _rmse(self, predictions, ground_truth):
        return np.sum((predictions - ground_truth) ** 2) / len(predictions)

    def _r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _gradient_descent(self, x_train, y_train, weights):
        loss_history = []
        
        initial_loss = self._least_squares_loss(x_train, y_train, weights)
        loss_history.append(initial_loss)

        for i in range(self.num_iterations):
            y_predicted = x_train.dot(weights)

            difference = y_predicted - y_train

            gradient = x_train.T.dot(difference) / x_train.shape[0]

            weights = weights - self.step_size * gradient

            updated_loss = self._least_squares_loss(x_train, y_train, weights)

            loss_history.append(updated_loss)
            
            if np.abs(loss_history[-2] - loss_history[-1]) <= self.convergence_threshold:
                break

        return weights, loss_history

    def find_optimal_hyperparams(self):
        best_rmse = np.inf
        self.all_hyperparam_tests = []

        for step_size in [0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]:
            for num_iterations in [20000, 40000, 50000, 80000, 120000]:

                rmse, r2, y_pred, time_diff = self.predict(step_size, num_iterations)

                if rmse < best_rmse:
                    best_rmse = rmse

                    self.optimal_hyperparams['step_size'] = step_size
                    self.optimal_hyperparams['num_iterations'] = num_iterations
                    self.eval_values['RMSE'] = rmse
                    self.eval_values['R2'] = self._r2_score(self.y_test, y_pred)
                    self.eval_values['Time'] = time_diff

                self.all_hyperparam_tests.append((step_size, num_iterations, rmse))

    def predict(self, step_size, num_iterations):
        self.reset_init_variables()
        self.step_size = step_size
        self.num_iterations = num_iterations

        start = time.time()
        w_opt, loss_history = self._gradient_descent(self.x_train, self.y_train, self.weights)
        y_pred = np.dot(self.x_test, w_opt)
        end = time.time()

        rmse = self._rmse(y_pred, self.y_test)
        time_diff = end - start
        r2 = self._r2_score(self.y_test, y_pred)

        return rmse, r2, y_pred, time_diff

    def process_inputs(self, x_values):
        x = x_values[:]
        c = np.ones(x.shape[0])
        add_c = np.vstack((x.T, c))
        new_x = add_c.T
        return new_x









