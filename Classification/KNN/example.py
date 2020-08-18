import pandas as pd
from Classification.KNN.k_nearest_neighbours import KNearestNeighbours
from Classification.point import Point

KNearestNeighbours(Point.create_point_list(pd.read_csv("../Datasets/Iris_Dataset.csv"), "Species"), 3,
                   test_size_percentage=0.2)

KNearestNeighbours(Point.create_point_list(pd.read_csv("../Datasets/2Class_Points.csv"), "cat"), 3,
                   test_size_percentage=0.2)
