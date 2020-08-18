import numpy as np


class LinearPerceptron:
    dimensions = 0
    x = []
    b = 0
    w = np.array

    def __init__(self, n_axes, points):
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
        print("Point: " + str(point.coordinates))
        print("Perceptron: " + str(self.w) + " - " + str(self.b))
        print("Classification: " + str(np.dot(self.w, point.coordinates) + self.b))
        print("Actual class: " + str(point.category))
        if (np.dot(self.w, point.coordinates) + self.b) * point.category > 0:
            print("Point " + str(point.coordinates) + " was classified correctly")
        else:
            print("Point " + str(point.coordinates) + " was misclassified")
        print()
        return np.dot(self.w, point.coordinates) + self.b


class Point:
    category = 0
    coordinates = np.array

    def __init__(self, category, coords):
        self.category = category
        self.coordinates = np.array(coords)
        self.coordinates.reshape(-1, 1)


x1 = Point(1, [-1, 1])
x2 = Point(-1, [0, -1])
x3 = Point(1, [10, 1])
perceptron = LinearPerceptron(2, [x1, x2, x3])


