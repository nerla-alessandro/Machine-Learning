import pandas as pd
from regression_tree import RegressionTree

dataset = pd.read_csv("../Datasets/winequality-red.csv")
model =  RegressionTree(dataset, ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", \
                                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", \
                                  "alcohol"], ["quality"], 3, 0.8)
