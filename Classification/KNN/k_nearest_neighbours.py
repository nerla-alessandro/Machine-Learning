from Classification.point import Point
import numpy as np
import random


class KNearestNeighbours:
    k = 0
    points = []

    def __init__(self, points: list, k: int, test_size_percentage=0):
        """
        Creates a KNN Model
        
        :param points: List of labelled points 
        :param k: Neighbours that vote on the class
        """
        if k > len(points):
            self.k = len(points)
        else:
            self.k = k

        if test_size_percentage > 0:
            random.shuffle(points)
            test_size = round(test_size_percentage * len(points))
            test = points[-test_size:]
            self.points = points[:-test_size]
            print("Train Set Size: " + str(len(self.points)) + " || Test Set Size: " + str(len(test)))
            if len(test) > 0 and len(test) < len(points):
                print("Accuracy: " + str(self.score(test)))
            else:
                print("ERROR: Invalid Test Set Size")
        else:
            self.points = points

    def score(self, points: []):
        n = len(points)
        hits = 0.0
        for p in points:
            if self.classify(p) == p.category:
                hits += 1.0
        return hits / n

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
