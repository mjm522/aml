#!/usr/bin/env python

import numpy as np
import pybullet as pb
import rospy
import time
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config
# from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_robot.bullet.bullet_sawyer import BulletSawyerArm
from aml_io.io_tools import get_aml_package_path, get_abs_path


def main():

    rospy.init_node('poke_box', anonymous=True)

    timeStep = 0.01
    
    phys_id = pb.connect(pb.SHARED_MEMORY)

    
    if (phys_id<0):
        phys_id = pb.connect(pb.GUI)
    
    pb.resetSimulation()
    
    pb.setTimeStep(timeStep)
    
    world = pb.loadURDF(pih_world_config['world_path'],[0,0,-.98])

    # peg = pb.loadURDF(pih_world_config['peg_path'])

    pb.setGravity(0., 0.,-10.)

    # hole = pb.loadURDF(pih_world_config['hole_path'], useFixedBase=True)

    catkin_ws_src_path = get_abs_path(get_aml_package_path()+'/../')
    print "catkin_ws_src_path:", catkin_ws_src_path
    manipulator = pb.loadURDF(catkin_ws_src_path + "/sawyer_robot/sawyer_description/urdf/sawyer.urdf", useFixedBase=True)
    pb.resetBasePositionAndOrientation(manipulator,[0,0,0],[0,0,0,1])
    # motors = [n for n in range(pb.getNumJoints(manipulator))]

    sawyerArm = BulletSawyerArm(manipulator)

    sawyerArm.untuck_arm()

    # print pb.getNumJoints(manipulator), "HRERERERER"
    # print sawyerArm._id, sawyerArm._nq, sawyerArm._ee_link_idx, "HERER"
    # k = np.array([1,1,1])

    # velocity_array = k*np.array([0, -0.05, -0.05])

    # pb.setJointMotorControlArray(manipulator, motors,controlMode=pb.VELOCITY_CONTROL, targetVelocities=np.asarray(velocity_array), forces=[500 for n in range(len(velocity_array))])

    # pm = PIHWorld(world_id=world, peg_id=peg, hole_id=hole, robot_id=manipulator, gains = np.asarray(k), config=pih_world_config)

    # pm.run()

    rate = rospy.Rate(100)

    pb.setRealTimeSimulation(0)


    # import time

    # time.sleep(1)
    # sawyerArm.exec_velocity_cmd([0.5,0,0,0,0,0,0])

    # rospy.on_shutdown(self.on_shutdown)

    # self._record_sample.start_record(task_action=pushes[idx])

    while not rospy.is_shutdown():

        # self.get_force_torque_details()

        # self.step()
        sawyerArm._update_state()

        print sawyerArm.get_ee_velocity()

        pb.stepSimulation()

        rate.sleep()

    pb.setRealTimeSimulation(1)

if __name__ == "__main__":    
    main()


# import pybullet as p
# # import time
# import math
# from datetime import datetime

# clid = p.connect(p.SHARED_MEMORY)
# if (clid<0):
#     p.connect(p.GUI)
# # p.loadURDF("plane.urdf",[0,0,-.98])

# # world = p.loadURDF(pih_world_config['world_path'])

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
# sawyerId = p.loadURDF("/home/saif/ros_ws/baxter_ws/src/sawyer_robot/sawyer_description/urdf/sawyer2.urdf",[0,0,0])
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
# p.resetBasePositionAndOrientation(sawyerId,[0,0,0],[0,0,0,1])

# #bad, get it from name! sawyerEndEffectorIndex = 18
# sawyerEndEffectorIndex = 16
# numJoints = p.getNumJoints(sawyerId)
# # print 'Num joints', numJoints

# #joint damping coefficents
# # jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]

# p.setGravity(0,0,0)
# t=0.
# prevPose=[0,0,0]
# prevPose1=[0,0,0]
# hasPrevPose = 0

# useRealTimeSimulation = 0
# p.setRealTimeSimulation(useRealTimeSimulation)
# #trailDuration is duration (in seconds) after debug lines will be removed automatically
# #use 0 for no-removal
# trailDuration = 15
    
# while 1:
#     if (useRealTimeSimulation):
#         dt = datetime.now()
#         t = (dt.second/60.)*2.*math.pi
#         print (t)
#     else:
#         t=t+0.01
#         time.sleep(0.01)
    
#     for i in range (1):
#         pos = [1.0,0.2*math.cos(t),0.+0.2*math.sin(t)]
#         jointPoses = p.calculateInverseKinematics(sawyerId,sawyerEndEffectorIndex,pos)

#         #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
#         for i in range (numJoints):
#             jointInfo = p.getJointInfo(sawyerId, i)
#             # print 'info', jointInfo
#             qIndex = jointInfo[3]
#             if qIndex > -1:
#                 p.resetJointState(sawyerId,i,jointPoses[qIndex-7])

#     ls = p.getLinkState(sawyerId,sawyerEndEffectorIndex)
#     if (hasPrevPose):
#         p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
#         p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
#     prevPose=pos
#     prevPose1=ls[4]
#     hasPrevPose = 1     

#         