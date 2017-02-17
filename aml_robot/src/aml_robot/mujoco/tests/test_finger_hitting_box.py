import os
import rospy
import random
import numpy as np
import quaternion

from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer
from aml_robot.mujoco.push_world.config import config_push_world
from aml_robot.utilities.utilities import convert_pose_to_euler_tranform


class PushMachine():

    def __init__(self, robot_interface):

        self._robot = robot_interface

    def choose_push_location(self):

        length_div2  = config_push_world['box_type']['length']/2
        breadth_div2 = config_push_world['box_type']['breadth']/2

        pre_push_offsets = config_push_world['pre_push_offsets']

        if random.uniform(0.,1.) > 0.5:
            weight = 1.
        else:
            weight = -1.

        if random.uniform(0.,1.) > 0.5:

            x_box = random.uniform(-length_div2, length_div2)   # w.r.t box frame
            pre_push_wrt_box = np.array([x_box, weight*pre_push_offsets[1], 0.,1.])
            
        else:

            y_box = random.uniform(-breadth_div2,breadth_div2) # w.r.t box frame
            pre_push_wrt_box = np.array([weight*pre_push_offsets[0], y_box, 0.,1.])

        box_pose = self._robot._model.data.qpos.flatten()[:7]

        push_location_wrt_world = np.dot(convert_pose_to_euler_tranform(box_pose), pre_push_wrt_box[:,None])

        self._robot.set_qpos(np.vstack([self._robot._model.data.qpos[0:7], 
                                        push_location_wrt_world[0:3], 
                                        self._robot._model.data.qpos[10:]]))

        return push_location_wrt_world[0:3]


    def push_box(self, force_mag=2., pre_push_location=None):

        box_pos =  self._robot._model.data.qpos.flatten()[0:3]

        finger_pos = self._robot._model.data.qpos.flatten()[7:10]

        force = force_mag*(box_pos-finger_pos)/np.linalg.norm(box_pos-finger_pos)

        qfrc_target = self._robot._model.applyFT(point=np.array([0., 0., 0.]), 
                                                 force=force, 
                                                 torque=np.zeros(3), body_name='Box')

        # print qfrc_target
        # self._robot._model.data.qfrc_applied = qfrc_target

        self._robot._model.step()

    
    def run(self):
        self.choose_push_location()
        
        self.push_box()





if __name__ == "__main__":
    
    rospy.init_node('poke_box', anonymous=True)

    robot_interface = MujocoRobot(xml_path=config_push_world['model_name'])

    viewer = MujocoViewer(mujoco_robot=robot_interface, width=config_push_world['image_width'], height=config_push_world['image_height'])

    viewer.configure(cam_pos=config_push_world['camera_pos'])

    robot_interface._configure(viewer=viewer, p_start_idx=7, p_end_idx=14, v_start_idx=6, v_end_idx=12)

    push_machine = PushMachine(robot_interface)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        push_machine.run()

        viewer.loop()

        rate.sleep()