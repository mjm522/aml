#!/usr/bin/env python
import rospy
import socket
import numpy as np
from aml_io.log_utils import aml_logging


class FTSensor():

    def __init__(self, ip="10.0.11.189", port=50000, rate=300):

        self._logger = aml_logging.get_logger(__name__)

        self._udp_ip = ip

        self._udp_port = port

        self._ft_reading = None

        self._sock = socket.socket(socket.AF_INET, # Internet
                                   socket.SOCK_DGRAM) # UDP

        self._sock.bind((self._udp_ip, self._udp_port))

        update_period = rospy.Duration(1.0/rate)
        
        self._sensor_callback = rospy.Timer(update_period, self.update)


    def update(self, event):

        try:
            data, addr = self._sock.recvfrom(1024) # buffer size is 1024 bytes
        except Exception as e:
            data = None

        # if data is not None:
        #     data = np.fromstring(data)

        self._ft_reading = data

    def ft_reading(self):

        return self._ft_reading
