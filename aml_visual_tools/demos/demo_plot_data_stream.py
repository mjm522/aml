import numpy as np
# import multiprocessing
# from threading import Thread
from aml_visual_tools.plot_data_stream import PlotDataStream



def main():

    plotter = PlotDataStream(plot_title="reward_plot", plot_size=None, max_plot_length=200)

    # thread = Thread(target=)
    # thread.start()

    # simulate = multiprocessing.Process(None, plotter.update_plot)
    # simulate.start()

    while True:

        plotter.add_data(np.random.randn(1))
        # plotter.update_plot()


if __name__ == '__main__':
    main()