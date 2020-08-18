import numpy as np
import pandas as pd


class LinearPerceptron:
    dimensions = 0
    x = []
    b = 0
    w = np.array

    def __init__(self, n_axes, points):
        """
        Initialises the perceptron and trains it on the given points \n
        WARNING: If the points are not linearly separable no solution will be found

        :param n_axes: Perceptron Hyperplane Dimensions
        :param points: Point objects to train the perceptron on
        """
        self.dimensions = n_axes
        weight_builder = []
        for i in range(self.dimensions):
            weight_builder += [0]
        self.w = np.array(weight_builder)
        self.x = points
        self.learn()

    def learn(self):
        correctly_classified_streak = 0
        i = 0
        while correctly_classified_streak < len(self.x):
            if i == len(self.x):
                i = 0
            if self.classify(self.x[i]) * self.x[i].category <= 0:
                print()
                self.b = self.b + self.x[i].category
                self.w = self.w + self.x[i].coordinates * self.x[i].category
                correctly_classified_streak = 0
            else:
                correctly_classified_streak += 1
            i += 1
        print("/// TRAINING HAS ENDED ///")
        print()

    def classify(self, point):
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


class Point:
    category = 0
    coordinates = np.array

    def __init__(self, category, coords):
        self.category = category
        self.coordinates = np.array(coords)
        self.coordinates.reshape(-1, 1)

    @staticmethod
    def create_point_list(dataframe: pd.DataFrame, category_variable):
        points = []
        coordinates = np.array(dataframe.drop(category_variable, axis=1)).tolist()
        categories = np.array(dataframe[category_variable]).tolist()
        for i in range(0, len(coordinates)):
            points.append(Point(category=categories[i], coords=coordinates[i]))
        return points


perceptron = LinearPerceptron(2, Point.create_point_list(pd.read_csv("points.csv"), "cat"))
T1 = Point(-1, [1, 3])
T2 = Point(1, [8, 3])
perceptron.classify(T1)
perceptron.classify(T2)
