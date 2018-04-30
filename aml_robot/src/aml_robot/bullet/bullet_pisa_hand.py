import roslib

roslib.load_manifest('aml_robot')

# import rospy

import numpy as np
import quaternion
import pybullet as pb

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot_hand import BulletRobotHand
from aml_robot.robot_interface import RobotInterface
from aml_io.io_tools import get_file_path, get_aml_package_path

from aml_robot.bullet.config import config_hand_world

class BulletPisaHand(RobotInterface):

    def __init__(self, limb="right", on_state_callback=None):
        self._ready = False

        self._phys_id = pb.connect(pb.SHARED_MEMORY)

        if (self._phys_id < 0):
            self._phys_id = pb.connect(pb.UDP, "127.0.0.1")

        pb.resetSimulation()

        pb.setGravity(0.0, 0.0 ,0.0)
        pb.setRealTimeSimulation(1)
        pb.setTimeStep(0.01)

        models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        sawyer_path = get_file_path('pisa_hand_right.urdf', models_path)
        robot_id = pb.loadURDF(sawyer_path, useFixedBase=True)

        self._config = config_hand_world
        self._bullet_robot = BulletRobotHand(robot_id=robot_id, config = config_hand_world)  # hardcoded from the sawyer urdf

        self._limb = limb

        if self._config['use_synergy']:

            self._joint_names = self._config['synergy_joints']
        else:
            self._joint_names = self._bullet_robot.joint_names(None)

        # all_joint_dict = self._bullet_robot.get_joint_dict()
        # self._joints = self._bullet_robot.get_all_joints()


        self._nq = self._bullet_robot.n_joints()
        self._nu = self._bullet_robot.n_cmd()

        lower_limits = self._bullet_robot.get_joint_limits()['lower']#[self._joints]
        upper_limits = self._bullet_robot.get_joint_limits()['upper']#[self._joints]

        self._jnt_limits = [{'lower': x[0], 'upper': x[1]} for x in zip(lower_limits,upper_limits)]

        self._tuck = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._untuck = self._tuck

        self._ready = True

    def _on_joint_states(self, msg):
        print 'NOT IMPLEMENTED'

    def exec_position_cmd(self, cmd):
        self._bullet_robot.set_joint_positions(cmd)

    def exec_position_cmd_delta(self, cmd):
        self._bullet_robot.set_joint_positions(self.angles() + cmd)

    def move_to_joint_position(self, cmd):
        self._bullet_robot.set_joint_positions(cmd)

    def move_to_joint_pos_delta(self, cmd):
        self._bullet_robot.set_joint_positions(self.angles() + cmd)

    def exec_velocity_cmd(self, cmd):
        self._bullet_robot.set_joint_velocities(cmd)

    def exec_torque_cmd(self, cmd):
        self._bullet_robot.set_joint_torques(cmd)

    def forward_kinematics(self, joint_angles=None):
        print 'NOT IMPLEMENTED'

    def inverse_kinematics(self, position, orientation=None):
        return self._bullet_robot.inverse_kinematics(position, orientation)#[self._joints]

    def ee_pose(self):
        return self._bullet_robot.ee_pose()

    def set_sampling_rate(self, sampling_rate=100):
        pass

    def untuck(self):
        pass
        # self.exec_position_cmd(self._tuck)

    def tuck(self):
        pass
        # self.exec_position_cmd(self._tuck)

    def _update_state(self):
        pass

    def cartesian_velocity(self, joint_deltas=None):
        pass

    def jacobian(self, joint_angles=None):

        return None
        # return self._bullet_robot.jacobian(joint_angles)

    def inertia(self, joint_angles=None):
        return self._bullet_robot.inertia(joint_angles)

    def q_mean(self):
        return self._bullet_robot.q_mean()

    def state(self):
        joint_angles = self.angles()
        joint_velocities = self.joint_velocities()
        joint_efforts = self.joint_efforts()

        state = {}
        state['position'] = joint_angles
        state['velocity'] = joint_velocities
        state['effort'] = joint_efforts
        state['jacobian'] = None#self.jacobian(None)[:, 1:]
        state['inertia'] = None#self.inertia(None)[1:, 1:]

        state['ee_point'], state['ee_ori'] = self.ee_pose()

        state['ee_vel'], state['ee_omg'] = self.ee_velocity()

        # state['gripper_state'] = self.gripper_state()


        return state

    def angles(self):
        return self._bullet_robot.angles()#[self._joints]

    def joint_velocities(self):
        return self._bullet_robot.joint_velocities()#[self._joints]

    def joint_efforts(self):
        return self._bullet_robot.joint_efforts()#[self._joints]

    def ee_velocity(self, numerical=False):
        return self._bullet_robot.ee_velocity(numerical)

    def n_cmd(self):
        return self._nu

    def n_joints(self):
        return self._nq

    def joint_names(self):
        return self._joint_names

    def joint_limits(self):
        return self._jnt_limits
