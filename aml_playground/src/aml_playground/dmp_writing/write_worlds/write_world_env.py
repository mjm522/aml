import os
import numpy as np
import pybullet as pb
from config import WRITE_ENV_CONFIG
from aml_robot.bullet.bullet_sawyer import BulletSawyerArm

class WriteEnv():

    def __init__(self, config=WRITE_ENV_CONFIG):

        self._cid = pb.connect(pb.SHARED_MEMORY)

        if (self._cid<0):
            self._cid = pb.connect(pb.GUI)
        
        self._config = config

        self._start_js = [-0.38079164,  0.62356653, -0.34625824, -1.01102594,  0.29616319,  1.90879249, -1.46761734] #joint angles

        self._joints = [3, 8, 9, 10, 11, 13, 16] #joint index values

        self.reset()


    def reset(self):

        pb.resetSimulation()
        
        pb.setTimeStep(self._config['dt'])

        pb.setGravity(0., 0.,-9.81)

        sawyer_path = os.path.join(self._config['urdf_root_path'],"sawyer/sawyer2_with_peg.urdf")

        kinematics_des = os.path.join(self._config['urdf_root_path'],"sawyer/sawyer.urdf")

        self._sawyer = BulletSawyerArm(phys_id=self._cid, sawyer_path=sawyer_path, kinematics=kinematics_des) #sawyer model

        self._table_id = pb.loadURDF(os.path.join(self._config['urdf_root_path'], "table.urdf"), useFixedBase=True, #table model
                                                 globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [1.0,-0.2,-0.12], [0, 0, -0.707, 0.707], physicsClientId=self._cid)


        self._sawyer._bullet_robot.set_joint_angles(joint_angles=self._start_js, joint_indices=self._joints)


        pb.setRealTimeSimulation(1)

        self._sawyer.step()


    def step(self):
        pb.stepSimulation()



