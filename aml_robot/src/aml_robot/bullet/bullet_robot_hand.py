import sys
import numpy as np
import quaternion
import pybullet as pb
from aml_robot.bullet.bullet_robot import BulletRobot
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


class BulletRobotHand(BulletRobot):
    def __init__(self, robot_id, config):

        BulletRobot.__init__(self, robot_id, config)

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

        self._synergy_joints = [self.get_joint_by_name(jm) for jm in self._config['synergy_joints']]

        self._joint_map = {"thumb": self._thumb_joints,
                           "index": self._index_joints,
                           "middle": self._middle_joints,
                           "ring": self._ring_joints,
                           "little": self._little_joints,
                           "synergy": self._synergy_joints}

        self._joint_name_map = {"thumb": self._config['thumb_joints'],
                               "index": self._config['index_joints'],
                               "middle": self._config['middle_joints'],
                               "ring": self._config['ring_joints'],
                               "little": self._config['little_joints'],
                               "synergy": self._config['synergy_joints']}

        # if self._config['use_synergy']:
        #     self._joint_name_map = {}
        #     self._joint_name_map['synergy_joints'] = self._config['synergy_joints']
        #     self._all_joint_names = self._joint_name_map['synergy_joints']

        self._all_joint_names = []
        for finger_name in self._config["finger_order"]:
            self._all_joint_names += self._joint_name_map[finger_name]

        if self._config['use_synergy']:

            self._joint_names = self._config['synergy_joints']
        else:
            self._joint_names = self._all_joint_names

        if self._config['use_synergy'] and not self._config['map_synergy']:
            self._all_joint_names = self._joint_names


        self._all_joints = []
        for finger_name in self._config["finger_order"]:
            self._all_joints += self._joint_map[finger_name]
        self._all_joints = np.array(self._all_joints)

        self._controllable_joints = self._all_joints

        if self._config['use_synergy'] and not self._config['map_synergy']:
            self._controllable_joints = self._synergy_joints



        self._all_joint_dict = dict(zip(self._all_joint_names, self._all_joints))

        self._nfingers = len(self._config["finger_order"])
        # Hack for the right hand
        # self._all_joints.reverse()

        # self._joint_state = np.zeros(len(self._joints))

        self._nq = len(self._all_joints)
        self._nu = len(self._all_joint_names)


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



    def joint_names(self, finger_name = None):

        if self._config['use_synergy']:
            joint_names = self._joint_names
        else:
            joint_names = self._all_joint_names

        return self._joint_name_map.get(finger_name, joint_names)


    def get_all_joints(self):

        return self._all_joints


    def set_jnt_state(self, jnt_state):


        if len(jnt_state) < self.n_joints():
            raise Exception ("Incorrect number of joint state values given")

        else:
            for jnt_idx in self.get_all_joints():

                pb.resetJointState(self._id, jnt_idx, jnt_state[jnt_idx])


    def set_joint_velocities(self, cmd, joints=None):

        if self._config['use_synergy'] and self._config['map_synergy']:
            hand_cmd = cmd[-1]

            cmd = [hand_cmd]*self.n_joints()

        BulletRobot.set_joint_velocities(self, cmd, self.get_controllable_joints())

    def set_joint_torques(self, cmd, joints=None):

        if self._config['use_synergy'] and self._config['map_synergy']:
            hand_cmd = cmd[-1]

            cmd = [hand_cmd]*self.n_joints()

        BulletRobot.set_joint_torques(self, cmd, self.get_controllable_joints())

    def set_joint_positions_delta(self, cmd, joints=None, forces=None):

        if self._config['use_synergy'] and self._config['map_synergy']:
            hand_cmd = cmd[-1]

            cmd = [hand_cmd]*self.n_joints()

        BulletRobot.set_joint_positions_delta(self, cmd, self.get_controllable_joints())

    def set_joint_positions(self, cmd, joints=None, forces=None):

        if self._config['use_synergy'] and self._config['map_synergy']:
            hand_cmd = cmd[-1]

            cmd = [hand_cmd]*self.n_joints()

        BulletRobot.set_joint_positions(self, cmd, self.get_controllable_joints())




    def get_controllable_joints(self):

        return self._controllable_joints





