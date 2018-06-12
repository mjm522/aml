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

        self.reset()


    def reset(self):

        pb.resetSimulation()
        
        pb.setTimeStep(self._config['dt'])

        pb.setGravity(0., 0.,-9.81)

        self._table_id = pb.loadURDF(os.path.join(self._config['urdf_root_path'], "table.urdf"), useFixedBase=True, 
                                                 globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.5, 1,-0.5], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        sawyer_path = os.path.join(self._config['urdf_root_path'],"sawyer/sawyer2.urdf")

        kinematics_des = os.path.join(self._config['urdf_root_path'],"sawyer/sawyer.urdf")

        self._sawyer = BulletSawyerArm(phys_id=self._cid, sawyer_path=sawyer_path, kinematics=kinematics_des)

        pb.setRealTimeSimulation(1)


    def step(self):
        pb.stepSimulation()



