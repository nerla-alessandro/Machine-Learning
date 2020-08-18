import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import os.path
import seaborn


class PlotUtil:

    @staticmethod
    def plot(x_data, y_data, title="", x_label="", y_label=""):
        """
        Simplifies matplotlib's plotting function

        :param x_data: Input Data (X Axis Values)
        :param y_data: Input Data (Y Axis Values)
        :param title: Title of the Graph
        :param x_label: Label of the X Axis
        :param y_label: Label of the Y Axis
        """
        seaborn.scatterplot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()


class LinearRegressionModel:
    model = linear_model.TheilSenRegressor
    data = pd.DataFrame
    nan_val = ""
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    predictor_values = np.ndarray
    response_values = np.ndarray
    response_column = ""
    test_size = 0
    pickled_model = ""
    average_dict = {}
    standard_dev_dict = {}
    corr_coeff_dict = {}

    def __init__(self, dataset, predictor_variables, response_variables, test_size_percentage, nan_val=np.nan,
                 target_accuracy=0.51, auto_select_predictors=False, useful_corr_threshold=0.5):
        self.average_dict = {}
        self.standard_dev_dict = {}
        self.corr_coeff_dict = {}
        self.data = dataset
        self.response_column = response_variables
        self.nan_val = nan_val
        if auto_select_predictors:
            self.generate_dataset_data(useful_corr_threshold)
            self.corr_coeff_dict.pop(self.response_column[0])
            predictor_variables = self.corr_coeff_dict.keys()
            if len(predictor_variables) == 0:
                print("WARNING: DATASET HAS NO LINEAR CORRELATION")
                print()
                return
        self.adapt_dataset(predictor_variables, response_variables, nan_val=nan_val)
        self.pickled_model = "PickledModels/" + str(
            pd.util.hash_pandas_object(dataset[predictor_variables]).sum()) + ".pickle"
        if os.path.isfile(self.pickled_model):
            with open(self.pickled_model, "rb") as pickled_file:
                self.model = pickle.load(pickled_file)
            self.print_model_data()
        else:
            if test_size_percentage == 0:
                self.train_only()
                self.print_model_data(test_fit=False, train_fit=True)
            else:
                self.train(target_accuracy, test_size_percentage)
                self.print_model_data(train_fit=True)

    def adapt_dataset(self, predictor_variables, response_variables, nan_val):
        """
        Removes records with missing values and then saves the values from the requested columns

        :param predictor_variables: Tags of the columns containing predictor variables
        :param response_variables: Tags of the columns containing response variables
        :param nan_val: Value/Character used to represent null, NaN, NaT or missing values in a record
        """
        dataset = self.data.replace(to_replace=nan_val, value=np.nan)
        dataset = dataset.dropna()
        self.predictor_values = np.array(dataset[predictor_variables])
        self.response_values = np.array(dataset[response_variables])

    def score(self, x, y):
        """
        Calculates the accuracy of the model's predictions of the response variable using the predictor variable and
        returns it

        :param x: Pandas DataFrame containing values of predictor variables
        :param y: Pandas DataFrame containing values of response variables
        :return: Coefficient of Determination (R^2) calculated as (1 - Residual Sum of Squares / Total Sum of Squares)
        """
        x_arr = np.array(x)
        y_arr = np.array(y)
        if x_arr.ndim == 1:
            x_arr.shape = (len(x_arr), 1)
        return self.model.score(x_arr, y_arr)

    def train_only(self):
        """
        Trains the model using the whole dataset
        """
        linear = linear_model.LinearRegression()
        self.x_train, self.y_train = self.predictor_values, self.response_values
        linear.fit(self.predictor_values, self.response_values)
        self.model = linear
        with open(self.pickled_model, "wb") as f:
            pickle.dump(self.model, f)

    def train(self, target_accuracy, test_size_percentage):
        """
        Trains the model using part of the dataset as training values and part as test values

        :param target_accuracy: Accuracy on test values that the model has to obtain before ending training
        :param test_size_percentage: Percentage of the dataset to use as test values
        """
        max_acc = 0
        while max_acc < target_accuracy:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                model_selection.train_test_split(self.predictor_values, self.response_values,
                                                 test_size=test_size_percentage)
            linear = linear_model.LinearRegression()
            linear.fit(self.x_train, self.y_train)
            self.model = linear
            acc = self.model.score(self.x_test, self.y_test)
            if acc > max_acc:
                max_acc = acc
                with open(self.pickled_model, "wb") as f:
                    pickle.dump(self.model, f)

    def predict(self, predictor_values):
        """
        Given a list of predictor variables' values, predicts the response variable's value

        :param predictor_values: Predictor variables' values
        :return: Predicted value of the response variable
        """
        return self.model.predict(predictor_values)

    def print_model_data(self, train_fit=False, test_fit=True, coeff=False, intercept=False):
        if train_fit:
            print("Training Data Fit: " + str(self.model.score(self.x_train, self.y_train)))
        if test_fit:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                model_selection.train_test_split(self.predictor_values, self.response_values, test_size=0.9)
            print("Accuracy: " + str(self.model.score(self.x_test, self.y_test)))
        if coeff:
            print("Coefficient(s):" + str(self.model.coef_))
        if intercept:
            print("Intercept:" + str(self.model.intercept_))
        print()

    def generate_dataset_data(self, useful_corr_threshold):
        """
        For every variable in the dataset, generates the following: \n
        1- Mean [Every Variable] \n
        2- Standard Deviation [Every Variable] \n
        3- Pearson Correlation Coefficient [For coefficients that have an absolute value over a certain threshold]

        :param useful_corr_threshold: Correlation coefficients with an absolute value under this threshold are discarded
        """
        # averages
        for variable in self.data:
            average = 0.0
            n = 0
            for x in self.data[variable]:
                if x != self.nan_val:
                    try:
                        average += float(x)
                        n += 1
                    except ValueError:
                        pass
            if n != 0:
                average /= n
                self.average_dict[variable] = average
            else:
                self.average_dict[variable] = "N/A"

        # standard deviations
        for variable in self.data:
            standard_deviation = 0
            n = 0
            if isinstance(self.average_dict[variable], int) or isinstance(self.average_dict[variable], float):
                for x in self.data[variable]:
                    try:
                        standard_deviation += np.power(float(x) - self.average_dict[variable], 2)
                        n += 1
                    except ValueError:
                        pass
            if n != 0:
                standard_deviation /= n - 1
                standard_deviation = np.sqrt(standard_deviation)
                self.standard_dev_dict[variable] = standard_deviation
            else:
                self.standard_dev_dict[variable] = "N/A"

        # correlation coefficient
        for variable in self.data:
            correlation_coeff = 0
            n = 0
            if isinstance(self.standard_dev_dict[variable], int) or isinstance(self.standard_dev_dict[variable], float):
                for x in self.data[variable]:
                    try:
                        x_star = (float(x) - self.average_dict[variable]) / self.standard_dev_dict[variable]
                        y_star = (self.data[self.response_column[0]][n] - self.average_dict[self.response_column[0]]) \
                            / self.standard_dev_dict[self.response_column[0]]
                        correlation_coeff += (x_star * y_star)
                        n += 1
                    except ValueError:
                        pass
            if n != 0:
                correlation_coeff /= n
                self.corr_coeff_dict[variable] = correlation_coeff
            else:
                self.corr_coeff_dict[variable] = "N/A"

        # saves only useful variables
        correlation_coeff_dict_copy = self.corr_coeff_dict.copy()
        for corr_coeff in correlation_coeff_dict_copy:
            if self.corr_coeff_dict[corr_coeff] == "N/A":
                self.corr_coeff_dict.pop(corr_coeff)
            elif useful_corr_threshold > self.corr_coeff_dict[corr_coeff] > -useful_corr_threshold:
                self.corr_coeff_dict.pop(corr_coeff)

    def print_dataset_data(self):
        """
        Prints statistical data regarding the dataset: \n
        1- Mean
        2- Standard Deviation
        3- Pearson Correlation Coefficient
        """
        self.generate_dataset_data(useful_corr_threshold=0)
        print("Averages:")
        print(self.average_dict)
        print()
        print("Standard Deviations:")
        print(self.standard_dev_dict)
        print()
        print("Pearson Correlation Coefficients:")
        print(str(self.corr_coeff_dict))
        print()
