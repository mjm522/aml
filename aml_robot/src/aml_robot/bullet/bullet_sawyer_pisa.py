import roslib

roslib.load_manifest('aml_robot')

# import rospy

import numpy as np
import quaternion
import pybullet as pb

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot_hand import BulletRobotHand
from aml_robot.bullet.bullet_robot import BulletRobot
from aml_robot.robot_interface import RobotInterface
from aml_io.io_tools import get_file_path, get_aml_package_path

from aml_robot.sawyer_kinematics import sawyer_kinematics
from aml_robot.bullet.config import config_hand_world

class BulletSawyerPisa(RobotInterface):

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
        sawyer_path = get_file_path('sawyer2_with_pisa_hand.urdf', models_path)
        table_path = get_file_path("table.urdf", models_path)
        mug_path = get_file_path("mug.urdf", models_path)
        plane_path = get_file_path("plane.urdf", models_path)
        robot_id = pb.loadURDF(sawyer_path, useFixedBase=True)
        self._table_id = pb.loadURDF(table_path, useFixedBase=True, globalScaling=0.5)
        pb.resetBasePositionAndOrientation(self._table_id, [0.7, 0., 0.6], [0, 0, -0.707, 0.707])

        self._mug_id = pb.loadURDF(mug_path,  globalScaling=0.001)
        pb.resetBasePositionAndOrientation(self._mug_id, [0.65, 0., 0.665], [-0.707, 0.0, 0.0, 0.707])

        self._plane_id = pb.loadURDF(plane_path, useFixedBase=True)
        pb.resetBasePositionAndOrientation(self._mug_id, [0.65, 0., 0.665], [-0.707, 0.0, 0.0, 0.707])

        self._bullet_robot = BulletRobot(robot_id=robot_id, config = config_hand_world)
        self._bullet_robot_hand = BulletRobotHand(robot_id=robot_id, config = config_hand_world)  # hardcoded from the sawyer urdf


        self._bullet_robot.configure_default_pos([-0.100000, 0.000000, 1.0000], [0.000000, 0.000000, 0.000000, 1.000000])

        self.name = limb
        self._limb = limb

        self._joint_names = ['right_j%s' % (s,) for s in range(0, 7)]
        self._joint_names += self._bullet_robot_hand.joint_names()

        self._kinematics = sawyer_kinematics(self, description=sawyer_path)

        # all_joint_dict = self._bullet_robot.get_joint_dict()
        # self._joints = self._bullet_robot.get_all_joints()

        all_joint_dict = self._bullet_robot.get_joint_dict()
        self._joints = [all_joint_dict[joint_name] for joint_name in self._joint_names]

        self._nq = len(self._joint_names)
        self._nu = len(self._joint_names)

        lower_limits = self._bullet_robot.get_joint_limits()['lower'][self._joints]
        upper_limits = self._bullet_robot.get_joint_limits()['upper'][self._joints]

        self._jnt_limits = [{'lower': x[0], 'upper': x[1]} for x in zip(lower_limits,upper_limits)]

        # self._tuck = np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00,
        #                        0., 0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.])
        self._tuck = np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00,
                               1.5707889560579844, 0.786210883026469, 0.7880146387387372, 0.0004999999996493284, 0.4795928796782728, 0.6948884694870044, 0.7828197973076467, 0.00033187601344706225, 0.7839629628325961, 0.7854953738051564, 0.785411633485201, -0.05623798776832702, 0.7827861068048277, 0.7834708238660533, 0.7741387171672663, 0.10818102126910266, 0.7794819174352743, 0.7875821720282794, 0.7788698724995202])

        self._untuck = np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
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
        self.exec_position_cmd(self._untuck)

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

    def joint_limits(self):
        return self._jnt_limits
