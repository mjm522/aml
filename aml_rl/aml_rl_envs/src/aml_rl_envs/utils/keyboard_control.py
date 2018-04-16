import os
import time
import numpy as np
import pybullet as pb

class KeyBoardControl():

    def __init__(self, manipulator, demo_path=os.environ["AML_DATA"]+'/data/new_data'):
        """
        constructor of the class
        Args:
        manipulator of type sawyer/threelink/kuka
        """

        self._robot = manipulator
        self._robot_id = manipulator._robot_id
        self._start = None
        self._end = None
        self._eps = 0.00001

        if not os.path.exists(demo_path):
            os.makedirs(demo_path)

        self._dof = len(self._robot._jnt_indexs)

        self._cmd = np.zeros(self._dof)

        self._keys_dof = [49L, 50L, 51L, 52L, 53L, 54L, 55L]
        self._up_key = 65297L
        self._dn_key = 65298L


    def keyboard_handle(self):

        keyboard_events = pb.getKeyboardEvents()

        keys = keyboard_events.keys()

        for k in range(self._dof):

            if self._keys_dof[k] in keys:

                if self._up_key in keys:

                    self._cmd[k] += self._eps

                elif self._dn_key in keys:

                    self._cmd[k] -= self._eps


    def run(self):

        while True:

            self.keyboard_handle()

            self._robot.set_joint_state(self._cmd)