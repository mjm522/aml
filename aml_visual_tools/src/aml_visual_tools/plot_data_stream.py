import rospy
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from geometry_msgs.msg import (Point, Quaternion, Pose, Vector3, Transform, Wrench)


class PlotDataStream():
    def __init__(self, plot_title=None, plot_size=None, max_plot_length=20):
        self._data = deque([])
        self._max_plot_length = max_plot_length
        self._plot_colors = ['r', 'g', 'b', 'm', 'c', 'k']
        if plot_title is None:
            plot_title = "Data Stream"
        if plot_size is None:
            plot_size = (10,5)
        plt.figure(plot_title, plot_size)
        plt.ion()

    def add_data(self, data):
        self._data.append(data)
        if len(self._data) > self._max_plot_length:
            self._data.popleft()

    def update_plot(self):
        plt.clf()
        data = np.asarray(self._data)
        sub_plot_indx = data.shape[1]*100+11
        for k in range(data.shape[1]):
            plt.subplot(sub_plot_indx)
            sub_plot_indx += 1
            plt.plot(data[:,k], linewidth=3, color=self._plot_colors[k%len(self._plot_colors)])
        plt.pause(0.0001)
        plt.draw()


class VisualizeROStopic():
    
    def __init__(self, config):
        self._config = config
        self._plot_data_stream = PlotDataStream(config['plot_name'], config['figsize'])
        rospy.Subscriber(config['rostopic'], config['msg_type'], self.data_callback)

    def data_callback(self, data):
        for field in self._config['msg_fields']:
            data = getattr(data, field)

        #TODO write similar field selection for other ros geometry types
        if type(data) == Vector3:
            self._plot_data_stream.add_data(np.array([data.x, data.y, data.z]))


    def run(self):
        rospy.init_node("stream continous data")
        while not rospy.is_shutdown():
            self._plot_data_stream.update_plot()

