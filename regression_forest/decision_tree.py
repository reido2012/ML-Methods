import numpy as np


class DecisionTree:

    def __init__(self, features, labels, max_depth, in_random_forest=False):
        self.in_random_forest = in_random_forest
        self.m = self._set_number_of_randomly_selected_features(features.shape[1])
        self.trained_tree = self.build_tree(features, labels, max_depth)

    def _set_number_of_randomly_selected_features(self, feature_dimension):
        if self.in_random_forest and feature_dimension > 3:
            return int(feature_dimension/3)
        else:
            return int(feature_dimension)

    def build_tree(self, x, y, max_depth):
        # Check if either of the stopping conditions have been reached. If so generate a leaf node...
        if max_depth == 1 or (y == y[0]).all():
            # Generate a leaf node...
            return {'leaf': True, 'value': None if len(y) == 0 else np.mean(y)}

        else:
            move = self.find_split(x, y)
            left = self.build_tree(x[move['left_indices'], :], y[move['left_indices']], max_depth - 1)
            right = self.build_tree(x[move['right_indices'], :], y[move['right_indices']], max_depth - 1)

            return {'leaf': False,
                    'feature': move['feature'],
                    'split': move['split'],
                    'rss': move['rss'],
                    'left': left,
                    'right': right}

    def calculate_rss(self, data):
        """Calculates residual sum of square differences """
        return np.sum((data - np.mean(data))**2)

    def find_split(self, x, y):
        """Given a dataset and its target values, this finds the optimal combination
        of feature and split point that gives the maximum information gain."""

        # Best thus far, initialised to a dud that will be replaced immediately...
        # we want to minimize rss
        best = {'rss': np.inf}

        # Loop every possible split of every dimension...
        possible_splits = range(x.shape[1])

        # Do random subspace method
        if self.in_random_forest:
            possible_splits = np.random.choice(possible_splits, self.m)

        for i in possible_splits:
            for pos, split in enumerate(np.unique(x[:, i])):
                lhs = y[x[:, i] <= split]
                rhs = y[x[:, i] > split]

                rss = self.calculate_rss(lhs) + self.calculate_rss(rhs)

                left_indices = np.nonzero(x[:, i] <= split)[0]
                right_indices = np.nonzero(x[:, i] > split)[0]

                if rss < best['rss']:
                    best = {'feature': i,
                            'split': split,
                            'rss': rss,
                            'left_indices': left_indices,
                            'right_indices': right_indices}
        return best

    def predict(self, samples):
        """Predicts class for every entry of a data matrix."""
        ret = np.empty(samples.shape[0], dtype=np.float32)
        ret.fill(-1)
        indices = np.arange(samples.shape[0])

        def traverse(node, indices):
            nonlocal samples
            nonlocal ret

            if node['leaf']:
                ret[indices] = node['value']

            else:
                going_left = samples[indices, node['feature']] <= node['split']
                left_indices = indices[going_left]
                right_indices = indices[np.logical_not(going_left)]

                if left_indices.shape[0] > 0:
                    traverse(node['left'], left_indices)

                if right_indices.shape[0] > 0:
                    traverse(node['right'], right_indices)

        traverse(self.trained_tree, indices)

        return ret
