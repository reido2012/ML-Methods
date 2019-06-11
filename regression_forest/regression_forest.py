import numpy as np
import time

from . import decision_tree


class RegressionForest:
    """
    Create N random sub-samples of our data set with replacement.
    Train estimators on each sample.
    Calculate the average prediction.
    """
    def __init__(self, train_features, train_labels, test_features, test_labels, num_estimators, max_tree_depth=np.inf):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.num_estimators = num_estimators
        self.max_tree_depth = max_tree_depth
        self.estimators = []
        self.bootstrap_draws = []
        self.optimal_hyperparams = {'num_estimators': None, 'max_tree_depth': None}
        self.eval_values = {'RMSE': None, 'R2': None, 'Time': None}
        self.all_hyperparam_tests = None

    def reset_init_variables(self):
        self.estimators = []
        self.bootstrap_draws = []

    def _r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _bagging(self):
        for _ in range(self.num_estimators):
            random_indexes = np.random.choice(len(self.train_features), len(self.train_features))
            draw = (self.train_features[random_indexes], self.train_labels[random_indexes], random_indexes)
            self.bootstrap_draws.append(draw)

    def train_forest(self):
        self._bagging()

        for i in range(self.num_estimators):
            bootstrap_features, bootstrap_labels, _ = self.bootstrap_draws[i]
            self.estimators.append(decision_tree.DecisionTree(bootstrap_features, bootstrap_labels, self.max_tree_depth,
                                                              in_random_forest=True))

    def _predict(self, test_features):
        self.train_forest()

        all_predictions = []

        for estimator in self.estimators:
            tree_prediction = estimator.predict(test_features)
            all_predictions.append(tree_prediction)

        all_predictions = np.array(all_predictions)
        regression_forest_predictions = np.mean(all_predictions, 0)

        return regression_forest_predictions

    def find_optimal_hyperparams(self):
        self.all_hyperparam_tests = []
        best_rmse = np.inf

        for num_estimators in [5, 10, 20, 40, 80]:
            for max_tree_depth in [10, 20, 50, 100]:

                rmse, r2, y_predictions, time_diff = self.predict(num_estimators, max_tree_depth)

                if rmse < best_rmse:
                    best_rmse = rmse

                    self.optimal_hyperparams['num_estimators'] = num_estimators
                    self.optimal_hyperparams['max_tree_depth'] = max_tree_depth
                    self.eval_values['RMSE'] = best_rmse
                    self.eval_values['R2'] = self._r2_score(self.test_labels, y_predictions)
                    self.eval_values['Time'] = time_diff

                self.all_hyperparam_tests.append((num_estimators, max_tree_depth, rmse))

    def predict(self, num_estimators, max_tree_depth):
        self.num_estimators = num_estimators
        self.max_tree_depth = max_tree_depth
        self.reset_init_variables()

        start = time.time()
        y_predictions = self._predict(self.test_features)
        end = time.time()
        time_diff = end - start

        rmse = self._rmse(y_predictions, self.test_labels)
        r2 = self._r2_score(self.test_labels, y_predictions)

        self.eval_values['RMSE'] = rmse
        self.eval_values['R2'] =  r2
        self.eval_values['Time'] = time_diff

        return rmse, r2, y_predictions, time_diff

    def _rmse(self, predictions, ground_truth):
        return np.sum((predictions - ground_truth) ** 2) / len(predictions)

    def get_loss(self, y_predicted, y_true):
        # Mean Squared Error
        return 1/len(y_predicted) * np.sum((y_predicted - y_true) ** 2)
