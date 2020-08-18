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
