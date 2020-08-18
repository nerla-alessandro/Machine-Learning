from Classification.point import Point
import numpy as np


class KNearestNeighbours:
    k = 0
    points = []

    def __init__(self, points: list, k: int):
        """
        Creates a KNN Model
        
        :param points: List of labelled points 
        :param k: 
        """
        self.k = k
        self.points = points

    def classify(self, point: Point):
        """
        Classifies a Point object based on the labelled points in the model using the KNN algorithm

        :param point: Point to classify
        :return: Predicted class
        """
        distance_list = []
        votes = {}
        for p in self.points:
            distance = 0
            for i in range(0, len(p.coordinates)):
                distance += np.square(p.coordinates[i] - point.coordinates[i])
            distance = np.sqrt(distance)
            distance_list.append((p, distance))
        distance_list.sort(key=lambda tup: tup[1])
        for i in range(0, self.k):
            if distance_list[i][0].category in votes:
                votes[distance_list[i][0].category] += 1
            else:
                votes[distance_list[i][0].category] = 1
        return max(votes, key=lambda key: votes[key])
