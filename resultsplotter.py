import matplotlib as plotter
import pandas as panda


def plot_history(path, max_x, interval):
    with open(path, mode='r') as file:
        history_dataframe = panda.read_json(file)
        history_dataframe.plot(use_index=True,
                               y=['loss', 'val_loss'],
                               xticks=range(0, max_x, interval),
                               title='Model loss',
                               xlabel='Epoch',
                               ylabel='Loss value')
        plotter.pyplot.show()
        history_dataframe.plot(use_index=True,
                               y=['accuracy', 'val_accuracy'],
                               xticks=range(0, max_x, interval),
                               title='Model accuracy',
                               xlabel='Epoch',
                               ylabel='Accuracy')
        plotter.pyplot.show()


def plot_models(df_dict, dimension, max_x):
    fig, axis = plotter.pyplot.subplots()
    fig_title = 'Results of models - ' +dimension
    for df in df_dict.keys():
        df_dict[df].plot(ax=axis,
                         use_index=True,
                         y=dimension,
                         title=fig_title,
                         xlabel='Epoch',
                         ylabel=dimension,
                         xticks=range(0, max_x, 5),)
    axis.legend(df_dict.keys())
    plotter.pyplot.show()

