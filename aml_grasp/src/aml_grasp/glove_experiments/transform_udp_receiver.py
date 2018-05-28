import socket
import numpy as np
from aml_math.quaternion_utils import rot2quat

class MocapTransformReceiver(object):

    UDP_IP = "192.168.1.69"
    UDP_PORT = 50000

    def __init__(self,udp_ip =  UDP_IP, udp_port = UDP_PORT):

        self._udp_ip = udp_ip
        self._udp_port = udp_port

        self._socket = socket.socket(socket.AF_INET, # Internet
                                     socket.SOCK_DGRAM) # UDP

        self._socket.bind((self._udp_ip, self._udp_port))


    def read_data(self):


        data, _ = self._socket.recvfrom(512)
        values = np.array([float(val) for val in data.split(' ')])

        pos = values[:3]

        if not np.any(np.isnan(values[3:])):
            quat = rot2quat(np.reshape(values[3:],(3,3)))

        else:
            quat = values[3:7]

        # transform = values.reshape(3,4)

        # print transform.shape
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        return pos, quat






# mocap_receiver = MocapTransformReceiver()
# #
# while True:
#     pos, rot = mocap_receiver.read_data()
#     # print transform[:][0].shape
#     # norms = [np.linalg.norm(transform[i][:]) for i in range(3)]
#     print pos, rot, np.linalg.norm(rot)