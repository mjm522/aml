#!/usr/bin/python

"""
This file is a socket based master plotter,
created to circumvent the problem of lack of thread safety of matplotlib
The users nees to run this script in a terminal and correpondingly 
import the client in the script from which plot has to be initiated
"""

import numpy as np
from multiprocessing.connection import Client
from multiprocessing.connection import Listener
from aml_visual_tools.plot_data_stream import PlotDataStream


class MasterPlotter(object):
    """
    This is the master class that acts as a listener socket to
    the plot stream class. This class continously listens to the 
    client and accepts a connection.
    """

    def __init__(self, port_num=6000, plot_title="master_plot", plot_size=None, max_plot_length=200):
        """
        Constructor of the class
        Args: 
        port_num: the port on the local machine
        plot_title: name of the plot window
        plot_size: size of the plot window; defualt is (10,5)
        max_plot_length: length of the plot window
        """

        self.setup_listener(port_num)
        self._plotter = PlotDataStream(plot_title=plot_title, plot_size=plot_size, max_plot_length=max_plot_length)


    def setup_listener(self, port_num):
        """
        The function that sets up listener.
        The port number is the port on the local machine
        """
        address = ('localhost', port_num)     
        self._listener = Listener(address, authkey='master_client_plotter')
        self._listener_conn = self._listener.accept()
        print 'Connection Accepted From', self._listener.last_accepted

    def listener_callback(self):
        """
        The function that constantly monitors the client
        This expects a numpy array in string format as the data input
        For a demo, please chek aml_visual_tools/demos/demo_plot_data_stream.py
        """

        msg = self._listener_conn.recv()

        # do something with msg
        if msg == 'close':
            #close the exiting connection
            self._listener_conn.close()
            #listen for more
            self.setup_listener()
        else:
            self._plotter.add_data(np.fromstring(msg))
            print self._plotter.data

    def run(self):
        """
        This function runs continously listening
        to any messages sent from the client
        """
        while True:
            print self._plotter._data
            self.listener_callback()
            self._plotter.update_plot()


class ClientPlotter(object):
    """
    The client class. This is the class
    that need to be importe in the fuctions that
    demands plots
    """

    def __init__(self, port_num=6000):
        """
        Constructor of the class
        Args:
        port number: the port on the local machine
        """

        self.setup_client(port_num)

    def setup_client(self, port_num):
        """
        The function that sets up the client
        """
        address = ('localhost', port_num)
        self._client = Client(address, authkey='master_client_plotter')

    def send_data_master(self, data):
        """
        The data that is added to the stack comes to this class
        how to use this class can be seen in the demo given in 
        aml_visual_tools/demos/demo_plot_data_stream.py
        """

        if isinstance(data, np.ndarray):
            self._client.send(data.tostring())
        
        elif isinstance(data, str):
            self._client.send(data)
        
        elif isinstance(data, float):
            self._client.send(np.array([data]).tostring())

        elif isinstance(data, list):
            self._client.send(np.asarray(data).tostring())



def main():

    mp = MasterPlotter()
    mp.run()


if __name__ == '__main__':
    main()


