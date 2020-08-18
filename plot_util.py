from matplotlib import pyplot as plt
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
