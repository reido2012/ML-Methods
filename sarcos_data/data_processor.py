import csv
import io
import os
import zipfile
import numpy as np


class DataProcessor:
    DIRNAME = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, split_ratio, num_data_points_to_use):
        self.data = self._load_sarcos_data()

        self.data = self.data[:num_data_points_to_use, :]
        self.split = int(self.data.shape[0] * split_ratio)
        self.training_features, self.training_labels = self._create_splits()
        self.test_features, self.test_labels = self._create_splits(train=False)

    def _load_sarcos_data(self):
        with zipfile.ZipFile(self.DIRNAME + '/sarcos_inv.zip') as zf:
            with zf.open('sarcos_inv.csv') as f:
                sf = io.TextIOWrapper(f)
                reader = csv.reader(sf)
                sarcos_data = []
                for row in reader:
                    sarcos_data.append([v for v in row])
                return np.array(sarcos_data, dtype=np.float32)

    def _create_splits(self, train=True):
        if train:
            # Features
            train_x = self.normalise_data(self.data[:self.split, :21])
            #Labels
            train_y = self.normalise_data(self.data[:self.split, 21])

            return train_x, train_y
        else:
            test_x = self.normalise_data(self.data[self.split:, :21])
            test_y = self.normalise_data(self.data[self.split:, 21])

            return test_x, test_y

    def normalise_data(self, x_unnormalised):
        b = np.mean(x_unnormalised)
        a = np.std(x_unnormalised)
        x_normalised = (x_unnormalised - b) / a
        return x_normalised