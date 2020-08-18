import numpy as np
import pandas as pd


class Point:
    category = 0
    coordinates = np.array

    def __init__(self, category, coords):
        self.category = category
        self.coordinates = np.array(coords)
        self.coordinates.reshape(-1, 1)

    @staticmethod
    def create_point_list(dataframe: pd.DataFrame, category_variable):
        """
        Creates a list of Point objects from a suitable Pandas Dataframe

        :param dataframe: Dataframe containing point data
        :param category_variable: Column of the dataframe containing the category (-1 or 1)
        :return: List of Point objects
        """
        points = []
        coordinates = np.array(dataframe.drop(category_variable, axis=1)).tolist()
        categories = np.array(dataframe[category_variable]).tolist()
        for i in range(0, len(coordinates)):
            points.append(Point(category=categories[i], coords=coordinates[i]))
        return points


class SingleLayerPerceptron:
    dimensions = 0
    __x = []
    b = 0
    w = np.array

    def __init__(self, n_axes: int, points: list):
        """
        Initialises the perceptron and trains it on the given points \n
        Points can be one of two classes: -1 or +1

        :param n_axes: Perceptron Hyperplane Dimensions
        :param points: Point objects to train the perceptron on
        """
        self.dimensions = n_axes
        weight_builder = []
        for i in range(self.dimensions):
            weight_builder += [0]
        self.w = np.array(weight_builder)
        self.__x = points
        self.__learn()

    def __learn(self):
        correctly_classified_streak = 0
        i = 0
        while correctly_classified_streak < len(self.__x):
            if i == len(self.__x):
                i = 0
            if self.classify(self.__x[i]) * self.__x[i].category <= 0:
                print()
                self.b = self.b + self.__x[i].category
                self.w = self.w + self.__x[i].coordinates * self.__x[i].category
                correctly_classified_streak = 0
            else:
                correctly_classified_streak += 1
            i += 1
        print("/// TRAINING HAS ENDED ///")
        print()

    def classify(self, point: Point):
        """
        Classifies a point based on the perceptron boundary

        :param point: Point to classify
        :return: Predicted class of the Point
        """
        classification = np.dot(self.w, point.coordinates) + self.b
        if classification != 0:
            classification = int(classification / np.abs(classification))
        print("Point: " + str(point.coordinates))
        print("Perceptron: w: " + str(self.w) + "  b: " + str(self.b))
        print("Classification: " + str(classification))
        print("Actual class: " + str(point.category))
        if (classification * point.category) > 0:
            print("Point " + str(point.coordinates) + " was classified correctly")
        else:
            print("Point " + str(point.coordinates) + " was misclassified")
        print()
        return classification
