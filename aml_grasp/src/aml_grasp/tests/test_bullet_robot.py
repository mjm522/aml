import numpy as np
import pybullet as pb

from aml_io.io_tools import get_file_path, get_aml_package_path
from matplotlib import pyplot as plt
from aml_io.log_utils import aml_logging
from aml_robot.bullet.bullet_robot import BulletRobot


phys_id = pb.connect(pb.SHARED_MEMORY)

if (phys_id<0):
    
    phys_id = pb.connect(pb.DIRECT)
    
pb.resetSimulation()
    
pb.setTimeStep(0.01)

models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
sawyer_path = get_file_path('sawyer2.urdf', models_path)
manipulator = pb.loadURDF(sawyer_path, useFixedBase=True)

robot = BulletRobot(manipulator)