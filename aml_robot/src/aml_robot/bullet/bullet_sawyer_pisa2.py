import roslib

roslib.load_manifest('aml_robot')

# import rospy

import numpy as np
import quaternion
import pybullet as pb

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot_hand import BulletRobotHand
from aml_robot.bullet.bullet_robot import BulletRobot
from aml_robot.bullet.bullet_sawyer import BulletSawyerArm
from aml_io.io_tools import get_file_path, get_aml_package_path

from aml_robot.sawyer_kinematics import sawyer_kinematics
from aml_robot.bullet.config import config_hand_world

class BulletSawyerPisa2(BulletSawyerArm):

    def __init__(self, limb="right", on_state_callback=None):

        BulletSawyerArm.__init__(self,limb,on_state_callback)

        self._ready = False

        models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')

        table_path = get_file_path("table.urdf", models_path)
        mug_path = get_file_path("mug.urdf", models_path)
        plane_path = get_file_path("plane.urdf", models_path)

        self._table_id = pb.loadURDF(table_path, useFixedBase=True, globalScaling=0.5)
        pb.resetBasePositionAndOrientation(self._table_id, [0.7, 0., 0.6], [0, 0, -0.707, 0.707])

        self._mug_id = pb.loadURDF(mug_path,  globalScaling=0.001)
        pb.resetBasePositionAndOrientation(self._mug_id, [0.65, 0., 0.665], [-0.707, 0.0, 0.0, 0.707])

        self._plane_id = pb.loadURDF(plane_path, useFixedBase=True)
        pb.resetBasePositionAndOrientation(self._mug_id, [0.65, 0., 0.665], [-0.707, 0.0, 0.0, 0.707])

        self._bullet_robot_hand = BulletRobotHand(robot_id=self._bullet_robot._id, config = config_hand_world)  # hardcoded from the sawyer urdf


        self._bullet_robot.configure_default_pos([-0.100000, 0.000000, 1.0000], [0.000000, 0.000000, 0.000000, 1.000000])

        self.name = limb
        self._limb = limb

        self._joint_names += self._bullet_robot_hand.joint_names()

        all_joint_dict = self._bullet_robot.get_joint_dict()
        self._joints = [all_joint_dict[joint_name] for joint_name in self._joint_names]

        self._nq = len(self._joint_names)
        self._nu = len(self._joint_names)

        self._kinematics = sawyer_kinematics(self, description=self._sawyer_path)

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


    def set_joint_angles(self, joint_angles):

        self._bullet_robot.set_joint_angles(joint_angles, self._joints)