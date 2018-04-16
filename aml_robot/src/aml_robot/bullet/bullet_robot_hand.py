import sys
import numpy as np
import quaternion
import pybullet as pb
from aml_robot.bullet.bullet_robot2 import BulletRobot2
# from utils import parse_state
# from glove_interface.config import default_glove_config as g_config

import atexit


def to_bullet_quat(q, flip_z=False):
    return [q.x, q.y, q.z, q.w]
    # if flip_z:
    #     return [q.x, q.y, -q.z, q.w]#np.flip(quaternion.as_float_array(q),0)
    # else:


def from_bullet_to_np_quat(array):
    return quaternion.quaternion(array[3], array[0], array[1], array[2])


def make_np_quat(array):
    return quaternion.quaternion(array[0], array[1], array[2], array[3])


class BulletRobotHand(BulletRobot2):
    def __init__(self, robot_id, config):

        BulletRobot2.__init__(self, robot_id, config)

        self._config = config

        # self._ori_offset = quaternion.from_euler_angles(*config['orientation_offset'])

        # self._pos = np.array([0.0, 0., 0.])
        # self._ori_quat = np.quaternion(1, 0, 0, 0)
        # self._start_pos = np.array([0.0, 0., 0.])
        # self._start_ori = quaternion.from_euler_angles(0, 1.570796327, 3.141592654)
        #
        # self.configure_default_pos(np.array([0.0, 0., 0.]), to_bullet_quat(self._ori_offset * self._ori_quat))

        self._thumb_joints = [self.get_joint_by_name(jm) for jm in self._config['thumb_joints']]

        self._index_joints = [self.get_joint_by_name(jm) for jm in self._config['index_joints']]

        self._middle_joints = [self.get_joint_by_name(jm) for jm in self._config['middle_joints']]

        self._ring_joints = [self.get_joint_by_name(jm) for jm in self._config['ring_joints']]

        self._little_joints = [self.get_joint_by_name(jm) for jm in self._config['little_joints']]

        self._joint_map = {"thumb": self._thumb_joints,
                           "index": self._index_joints,
                           "middle": self._middle_joints,
                           "ring": self._ring_joints,
                           "little": self._little_joints}

        self._joint_name_map = {"thumb": self._config['thumb_joints'],
                               "index": self._config['index_joints'],
                               "middle": self._config['middle_joints'],
                               "ring": self._config['ring_joints'],
                               "little": self._config['little_joints']}

        self._all_joint_names = []
        for finger_name in self._config["finger_order"]:
            self._all_joint_names += self._joint_name_map[finger_name]

        self._all_joints = []
        for finger_name in self._config["finger_order"]:
            self._all_joints += self._joint_map[finger_name]

        self._all_joint_dict = dict(zip(self._all_joint_names, self._all_joints))

        self._nfingers = len(self._config["finger_order"])
        # Hack for the right hand
        # self._all_joints.reverse()

        # self._joint_state = np.zeros(len(self._joints))

        self._nq = len(self._all_joints)
        self._nu = len(self._all_joints)

        self._joint_limits = self.get_joint_limits()

        self.add_static_debug_visual_elements()

        pb.getCameraImage(640, 480, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        atexit.register(self.on_shutdown)

    def add_static_debug_visual_elements(self):

        for jn in self._all_joints:
            pb.addUserDebugLine([0, 0, 0], [0, 0.0, 0.05], [1, 0, 0], parentObjectUniqueId=self._id,
                                parentLinkIndex=jn, lineWidth=0.05, lifeTime=0)
            pb.addUserDebugText("jnt_%d" % jn, [0, 0, -0.025], textColorRGB=[1, 0, 0], textSize=1.0,
                                parentObjectUniqueId=self._id, parentLinkIndex=jn)
    #
    # def get_joint_limits(self, joints):
    #
    #     joint_lims = [self.get_joint_limits(joint_idx) for joint_idx in joints]
    #     return dict(zip(joints, joint_lims))
    #



    def on_shutdown(self):

        pb.removeAllUserDebugItems()

        pb.resetSimulation()

        pb.disconnect()



    def joint_names(self, finger_name):
        return self._joint_name_map.get(finger_name, self._all_joint_names)


    def get_all_joints(self):

        return np.array(self._all_joints)








