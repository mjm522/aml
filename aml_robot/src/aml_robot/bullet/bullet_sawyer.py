import roslib

roslib.load_manifest('aml_robot')

# import rospy

import numpy as np
import quaternion
import pybullet as pb

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot2 import BulletRobot2
from aml_robot.robot_interface import RobotInterface
from aml_io.io_tools import get_file_path, get_aml_package_path


# from aml_visual_tools.load_aml_logo import load_aml_logo

class BulletSawyerArm(RobotInterface):

    def __init__(self, limb="right", on_state_callback=None):


        self._phys_id = pb.connect(pb.SHARED_MEMORY)

    
        if (self._phys_id<0):
            self._phys_id = pb.connect(pb.UDP,"127.0.0.1")

        pb.resetSimulation()

        pb.setGravity(0., 0.,0.0)
        pb.setTimeStep(0.001)
        pb.setRealTimeSimulation(1)


        # description_path = config['description_path']
        # extension = description_path.split('.')[-1]
        # if extension == "urdf":
        #     self._id = pb.loadURDF(config['description_path'])
        # elif extension == "xml":
        #     self._id = pb.loadMJCF(config['description_path'])[0]

        models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        sawyer_path = get_file_path('sawyer2.urdf', models_path)
        robot_id = pb.loadURDF(sawyer_path, useFixedBase=True)

        self._bullet_robot = BulletRobot2(robot_id=robot_id, ee_link_idx=16, ee_link_name="right_hand")  # hardcoded from the sawyer urdf

        self._limb = limb

        self._joint_names = ['right_j%s' % (s,) for s in range(0, 7)]

        all_joint_dict = self._bullet_robot.get_joint_dict()
        self._joints = [all_joint_dict[joint_name] for joint_name in self._joint_names]

        self._nq = len(self._joint_names)
        self._nu = len(self._joint_names)

        self._jnt_limits = [{'lower': -1.70167993878, 'upper': 1.70167993878},
                            {'lower': -2.147, 'upper': 1.047},
                            {'lower': -3.05417993878, 'upper': 3.05417993878},
                            {'lower': -0.05, 'upper': 2.618},
                            {'lower': -3.059, 'upper': 3.059},
                            {'lower': -1.57079632679, 'upper': 2.094},
                            {'lower': -3.059, 'upper': 3.059}]

        self._tuck = np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00])
        self._untuck = self._tuck

        self._ready = True



    def _on_joint_states(self, msg):
        print 'NOT IMPLEMENTED'

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



    def forward_kinematics(self, joint_angles=None):

        print 'NOT IMPLEMENTED'


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
        return self._bullet_robot.jacobian(joint_angles)

    def inertia(self, joint_angles=None):
        return self._bullet_robot.inertia(joint_angles)

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
