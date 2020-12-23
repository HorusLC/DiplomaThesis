import matplotlib as plotter
import pandas as panda


def plot_history(path, max_x, interval):
    with open(path, mode='r') as file:
        history_dataframe = panda.read_json(file)
        history_dataframe.plot(use_index=True,
                               y=['loss', 'val_loss'],
                               xticks=range(0, max_x, interval))
        plotter.pyplot.show()
        history_dataframe.plot(use_index=True,
                               y=['accuracy', 'val_accuracy'],
                               xticks=range(0, max_x, interval))
        plotter.pyplot.show()

