import roslib

roslib.load_manifest('aml_robot')

# import rospy

import numpy as np
import quaternion
import pybullet as pb

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot import BulletRobot
from aml_robot.robot_interface import RobotInterface
from aml_io.io_tools import get_file_path, get_aml_package_path
from aml_robot.sawyer_kinematics import sawyer_kinematics
from aml_robot.bullet.config import SAWYER_BULLET_CONFIG

# from aml_visual_tools.load_aml_logo import load_aml_logo

class BulletSawyerArm(RobotInterface):

    def __init__(self, limb="right", on_state_callback=None):

        self._ready = False

        self._phys_id = pb.connect(pb.SHARED_MEMORY)

    
        if (self._phys_id<0):
            self._phys_id = pb.connect(pb.UDP,"127.0.0.1")

        pb.resetSimulation()

        pb.setGravity(0.0, 0.0 ,0.0)
        pb.setRealTimeSimulation(1)
        pb.setTimeStep(0.01)


        # description_path = config['description_path']
        # extension = description_path.split('.')[-1]
        # if extension == "urdf":
        #     self._id = pb.loadURDF(config['description_path'])
        # elif extension == "xml":
        #     self._id = pb.loadMJCF(config['description_path'])[0]

        models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        self._sawyer_path = get_file_path('sawyer2_with_pisa_hand.urdf', models_path)#_with_pisa_hand
        robot_id = pb.loadURDF(self._sawyer_path, useFixedBase=True)

        self.name = limb

        self._joint_names = ['right_j%s' % (s,) for s in range(0, 7)]

        self._kinematics = sawyer_kinematics(self, description=self._sawyer_path)

        self._bullet_robot = BulletRobot(robot_id=robot_id, config = SAWYER_BULLET_CONFIG)  # hardcoded from the sawyer urdf

        self._limb = limb


        all_joint_dict = self._bullet_robot.get_joint_dict()
        self._joints = [all_joint_dict[joint_name] for joint_name in self._joint_names]

        self._nq = len(self._joint_names)
        self._nu = len(self._joint_names)

        # self._jnt_limits = [{'lower': -1.70167993878, 'upper': 1.70167993878},
        #                     {'lower': -2.147, 'upper': 1.047},
        #                     {'lower': -3.05417993878, 'upper': 3.05417993878},
        #                     {'lower': -0.05, 'upper': 2.618},
        #                     {'lower': -3.059, 'upper': 3.059},
        #                     {'lower': -1.57079632679, 'upper': 2.094},
        #                     {'lower': -3.059, 'upper': 3.059}]

        lower_limits = self._bullet_robot.get_joint_limits()['lower'][self._joints]
        upper_limits = self._bullet_robot.get_joint_limits()['upper'][self._joints]

        self._jnt_limits = [{'lower': x[0], 'upper': x[1]} for x in zip(lower_limits,upper_limits)]

        self._tuck = np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00])
        self._untuck = self._tuck

        self._ready = True


    def exec_position_cmd(self, cmd):
        self._bullet_robot.set_joint_positions(cmd, self._joints)

    def exec_position_cmd_delta(self, cmd):
        self._bullet_robot.set_joint_positions(self.angles() + cmd, self._joints)

    def move_to_joint_position(self, cmd):
        self._bullet_robot.set_joint_positions(cmd, self._joints)

    def move_to_joint_pos_delta(self, cmd):
        self._bullet_robot.set_joint_positions(self.angles() + cmd, self._joints)


    def exec_velocity_cmd(self, cmd):

        self._bullet_robot.set_joint_velocities(cmd, self._joints)

    def exec_torque_cmd(self, cmd):

        self._bullet_robot.set_joint_torques(cmd, self._joints)



    def forward_kinematics(self, joint_angles=None, ori_type='quat'):

        if joint_angles is None:

            argument = None

        else:

            argument = dict(zip(self.joint_names(), joint_angles))

        # combine the names and joint angles to a dictionary, that only is accepted by kdl
        pose = np.array(self._kinematics.forward_position_kinematics(argument))
        position = pose[0:3][:, None]  # senting as  column vector

        w = pose[6]
        x = pose[3]
        y = pose[4]
        z = pose[5]  # quarternions

        rotation = quaternion.quaternion(w, x, y, z)

        # formula for converting quarternion to rotation matrix

        if ori_type == 'mat':

            # rotation = np.array([[1.-2.*(y**2+z**2),    2.*(x*y-z*w),           2.*(x*z+y*w)],\
            #                      [2.*(x*y+z*w),         1.-2.*(x**2+z**2),      2.*(y*z-x*w)],\
            #                      [2.*(x*z-y*w),         2.*(y*z+x*w),           1.-2.*(x**2+y**2)]])

            rotation = quaternion.as_rotation_matrix(rotation)

        elif ori_type == 'eul':

            rotation = quaternion.as_euler_angles(rotation)
        elif ori_type == 'quat':
            pass

        return position, rotation


    def inverse_kinematics(self, position, orientation=None):
        return self._bullet_robot.inverse_kinematics(position, orientation)[self._joints]


    def ee_pose(self):
        return self._bullet_robot.ee_pose()


    def set_sampling_rate(self, sampling_rate=100):
        pass

    def untuck(self):
        self.exec_position_cmd(self._tuck)

    def tuck(self):
        self.exec_position_cmd(self._tuck)

    def _update_state(self):
        pass

    def cartesian_velocity(self, joint_deltas=None):
        pass

    def jacobian(self, joint_angles=None):

        # jacobian = self._bullet_robot.jacobian(joint_angles)
        #
        # return np.delete(jacobian, 1, 1)

        if joint_angles is None:

            argument = dict(zip(self.joint_names(), self.angles()))

        else:

            argument = dict(zip(self.joint_names(), joint_angles))
        # combine the names and joint angles to a dictionary, that only is accepted by kdl
        jacobian = np.array(self._kinematics.jacobian(argument))

        # print jacobian

        return jacobian

    def inertia(self, joint_angles=None):

        if joint_angles is None:
            argument = dict(zip(self.joint_names(), self.angles()))
        else:
            argument = dict(zip(self.joint_names(), joint_angles))

        return np.array(self._kinematics.inertia(argument))

    def q_mean(self):
        return self._bullet_robot.q_mean()[self._joints]

    def state(self):

        joint_angles = self.angles()
        joint_velocities = self.joint_velocities()
        joint_efforts = self.joint_efforts()

        state = {}
        state['position'] = joint_angles
        state['velocity'] = joint_velocities
        state['effort'] = joint_efforts
        state['jacobian'] = self.jacobian(None)
        state['inertia'] = self.inertia(None)

        state['ee_point'], state['ee_ori'] = self.ee_pose()

        state['ee_vel'], state['ee_omg'] = self.ee_velocity()

        # state['gripper_state'] = self.gripper_state()


        return state


    def angles(self):
        return self._bullet_robot.angles()[self._joints]


    def joint_limits(self):

        return self._jnt_limits

    def joint_velocities(self):
        return self._bullet_robot.joint_velocities()[self._joints]

    def joint_efforts(self):
        return self._bullet_robot.joint_efforts()[self._joints]

    def ee_velocity(self, numerical=False):
        return self._bullet_robot.ee_velocity(numerical)

    def n_cmd(self):
        return self._nu

    def n_joints(self):
        return self._nq

    def joint_names(self):
        return self._joint_names


    def set_gripper_speed(self, speed):

        pass

    def set_arm_speed(self, speed):

        pass

