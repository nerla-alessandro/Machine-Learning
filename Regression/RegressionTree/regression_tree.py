from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection


class RegressionTree:
    model = DecisionTreeRegressor

    def __init__(self, dataset, predictorVariables, responseVariable, max_depth, test_size_percentage):
        self.model = DecisionTreeRegressor(max_depth=max_depth)
        if test_size_percentage == 0:
            self.model.fit(dataset[predictorVariables], dataset[responseVariable])
        else:
            predictor_train, predictor_test, response_train, response_test = model_selection.train_test_split(dataset,
                                                                                                              test_size=test_size_percentage)
            self.model.fit(predictor_train, response_train)
            print(self.model.score(predictor_test, response_test))


