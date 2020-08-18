import pandas as pd
from Classification.Perceptron.single_layer_perceptron import SingleLayerPerceptron
from Classification.Perceptron.single_layer_perceptron import Point

perceptron = SingleLayerPerceptron(2, Point.create_point_list(pd.read_csv("Datasets/points.csv"), "cat"))
T1 = Point(-1, [1, 3])
T2 = Point(1, [8, 3])
perceptron.classify(T1)
perceptron.classify(T2)