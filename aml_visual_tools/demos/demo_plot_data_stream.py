import numpy as np
from aml_visual_tools.master_client_plotter import ClientPlotter


def main():

    client_plot = ClientPlotter()

    while True:

        client_plot.send_data_master(np.random.randn(1))

if __name__ == '__main__':
    main()