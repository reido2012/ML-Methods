class Evaluator:

    def __init__(self, models_to_evaluate):
        self.models_to_evaluate = models_to_evaluate
        self.results = {}
        self.prediction_results = {}

    def run_evaluation(self):

        for model in self.models_to_evaluate:

            evaluation_name = model.__class__.__name__
            print("Evaluating: " + evaluation_name)
            model.find_optimal_hyperparams()

            # Converts the values from list of tuples to a list of lists such that it creates a row in the table
            all_h_test_values = list(map(list, zip(*model.all_hyperparam_tests)))

            self.results[evaluation_name] = {'Analysis': model.eval_values, 'Hyperparams': model.optimal_hyperparams,
                                             'All_H_Values': all_h_test_values}
            print("Finished Evaluating: " + evaluation_name)

    def run_predictions(self):
        for model in self.models_to_evaluate:
            model_name = model.__class__.__name__
            print("Predicting: " + model_name)

            if model_name == 'GaussianProcess':
                rmse, r2, _, time_diff = model.predict(model.num_samples)

            if model_name == 'LinearRegression':
                rmse, r2, _, time_diff = model.predict(model.step_size, model.num_iterations)

            if model_name == 'RegressionForest':
                rmse, r2, _, time_diff = model.predict(model.num_estimators, model.max_tree_depth)

            if model_name == 'NearestNeighbour':
                rmse, r2, _, time_diff = model.predict(model.k)

            self.prediction_results[model_name] = {'Analysis': {'RMSE': rmse, 'R2': r2, 'Time': time_diff}}

            print("Finished Evaluating: " + model_name)

    def get_plottable_analysis(self, dictionary):
        RMSES = []
        R2 = []
        TIMES = []
        column_labels = []

        for method in dictionary.keys():
            rmse = dictionary[method]['Analysis']['RMSE']
            r2 = dictionary[method]['Analysis']['R2']
            time = dictionary[method]['Analysis']['Time']

            column_labels.append(method)
            RMSES.append(rmse)
            R2.append(r2)
            TIMES.append(time)

        row_labels = ["RMSE", "R2", "Time"]

        return [RMSES, R2, TIMES], column_labels, row_labels
