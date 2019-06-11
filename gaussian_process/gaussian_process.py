import numpy as np
import time


class GaussianProcess:
    def __init__(self, train_x, train_y, test_x, test_y, num_smamples_to_draw=100):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        self.num_samples = num_smamples_to_draw
        self.eval_values = {'RMSE': None, 'R2': None, 'Time': None}
        self.optimal_hyperparams = {'num_samples': None}
        self.all_hyperparam_tests = []

    def kernel(self, a, b):
        # Squared Exponential Kernel Function for GP
        square_distance = np.sum((a - b) ** 2)
        return np.exp(-0.5 * square_distance)

    def build_covariance(self, data_1, data_2):
        covariance_k = []

        for x_train in data_1:
            cov_row = []
            x_val = x_train
            for x_train_2 in data_2:
                x_val_2 = x_train_2

                cov_row.append(self.kernel(x_val, x_val_2))
            covariance_k.append(cov_row)

        return np.array(covariance_k)

    def train(self):
        # GP training means constructing K
        mean = np.zeros(len(self.train_x))
        covariance = self.build_covariance(self.train_x, self.train_x)

        return mean, covariance

    def draw_samples(self, mean, covariance, num_samples=1):
        # Every time we sample a D dimensional gaussian we get D points (function)
        all_samples = []

        # SVD gives better numerical stability than Cholesky Decomp (it was giving errors)
        num_dimensions = len(mean)

        for _ in range(0, num_samples):
            z = np.random.standard_normal(num_dimensions)

            [U, S, V] = np.linalg.svd(covariance)
            A = U * np.sqrt(S)

            all_samples.append(mean + np.dot(A, z))

        return all_samples

    def _predict(self, covariance_k):
        k = covariance_k
        k_star = self.build_covariance(self.train_x, self.test_x)
        k_star_star = self.build_covariance(self.test_x, self.test_x)

        k_star_inv = k_star.T.dot(np.linalg.pinv(k))
        mu_s = k_star_inv.dot(self.train_y)
        sigma_s = k_star_star - np.matmul(k_star_inv, k_star)
        return (mu_s, sigma_s)

    def _rmse(self, predictions, ground_truth):
        return np.sum((predictions - ground_truth) ** 2) / len(predictions)

    def _r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _full_rmse(self, all_samples):
        mean = np.mean(all_samples, 0)
        return self._rmse(self.test_y, mean)

    def find_optimal_hyperparams(self):
        self.all_hyperparam_tests = []
        best_error = np.inf

        for num_samples in [10, 20, 50, 100, 200, 400]:

            rmse, r2, new_all_samples, time_diff = self.predict(num_samples)

            if rmse < best_error:
                best_error = rmse
                self.optimal_hyperparams['num_samples'] = num_samples
                self.eval_values['RMSE'] = best_error
                self.eval_values['R2'] = r2
                self.eval_values['Time'] = time_diff

            self.all_hyperparam_tests.append((num_samples, rmse))

    def predict(self, num_samples):
        self.num_samples = num_samples

        start = time.time()
        mean, covariance = self.train()
        new_mean, new_covariance = self._predict(covariance)
        new_all_samples = self.draw_samples(new_mean, new_covariance, self.num_samples)
        end = time.time()

        rmse = self._rmse(np.mean(new_all_samples, 0), self.test_y)
        r2 = self._r2_score(self.test_y, np.mean(new_all_samples, 0))

        return rmse, r2, new_all_samples, end-start

    # def predict(self):
    #     start = time.time()
    #     mean, covariance = self.train()
    #     new_mean, new_covariance = self._predict(covariance)
    #     new_all_samples = self.draw_samples(new_mean, new_covariance, self.num_samples)
    #     end = time.time()
    #
    #     rmse = self._rmse(np.mean(new_all_samples, 0), self.test_y)
    #     r2 = self._r2_score(self.test_y, np.mean(new_all_samples, 0))
    #
    #     self.eval_values['RMSE'] = rmse
    #     self.eval_values['R2'] = r2
    #     self.eval_values['Time'] = end - start
    #
    #     return new_all_samples, (new_mean, new_covariance), (mean, covariance)





