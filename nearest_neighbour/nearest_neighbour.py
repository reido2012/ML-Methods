import numpy as np
import time


class NearestNeighbour:

    def __init__(self, train_inputs, train_outputs, test_inputs, test_outputs, k):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs
        self.optimal_hyperparams = {'k': None}
        self.eval_values = {'RMSE': None, 'R2': None, 'Time': None}
        self.all_hyperparam_tests = None
        self.k = k

    def predict(self, k):
        self.k = k
        test_predictions = []
        start = time.time()

        for new_sample in self.test_inputs:
            distances = []

            for idx, train_feature in enumerate(self.train_inputs):

                distance = self._manhattan_distance(train_feature, new_sample)
                distances.append((self.train_outputs[idx], distance))

            # Put the distances in order from lowest to highest
            ordered_distances = sorted(distances, key=(lambda x: x[1]))

            # Get the actual value for the target at those points
            top_k_outputs = [values[0] for values in ordered_distances[:self.k]]

            test_predictions.append(np.mean(top_k_outputs))
        end = time.time()

        y_predictions = np.array(test_predictions)
        error = self._rmse(y_predictions, self.test_outputs)
        r2 = self._r2_score(self.test_outputs, y_predictions)
        time_diff = end - start

        return error, r2,  y_predictions, time_diff

    def find_optimal_hyperparams(self):
        self.all_hyperparam_tests = []
        best_error = np.inf

        for k in range(1, 10):
            error, r2, y_predictions, time_diff = self.predict(k)

            if error < best_error:
                best_error = error
                self.optimal_hyperparams['k'] = k
                self.eval_values['RMSE'] = best_error
                self.eval_values['R2'] = r2
                self.eval_values['Time'] =time_diff

            self.all_hyperparam_tests.append((k, error))

    def _rmse(self, predictions, ground_truth):
        return np.sum((predictions - ground_truth) ** 2) / len(predictions)

    def _r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _manhattan_distance(self, x, y):
        return np.sum(np.abs(x-y))
