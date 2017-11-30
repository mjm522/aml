import cv2
import math
import numpy as np
import pybullet as pb
import rospy
import random
import time

from aml_robot.bullet.bullet_robot import BulletRobot
from aml_playground.peg_in_hole.pih_worlds.bullet.config import config_pih_world
from aml_data_collec_utils.record_sample import RecordSample

class BoxObject(BulletRobot):

    def __init__(self, box_id):

        super(BoxObject, self).__init__(robot_id=box_id, config=config_pih_world)

    def get_effect(self):
        p, q   = self.get_pos_ori()
        status = {}
        status['pos'] = p
        #all the files in package follows np.quaternion convention, that is
        # w,x,y,z while ros follows x,y,z,w convention
        status['ori'] = np.array([q[3],q[0],q[1],q[2]])

        return status

class PegHole():

    def __init__(self, hole_id, pos = [0. ,0. ,0.], ori = [0., 0., 0., 1]):

        self._pos = pos
        self._ori = ori
        self._id = hole_id
        self.set_default_pos(np.array(pos), np.array(ori))

    def set_default_pos(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._id, pos, ori)
        self._default_pos = pos
        self._default_ori = ori


class PIHWorld():

    def __init__(self, world_id, peg_id, hole_id, robot_id, config):

        self._peg      = BoxObject(box_id=peg_id)

        self._world_id = world_id

        self._hole = PegHole(hole_id, [0,0,1], [0, 0, -0.707, 0.707])

        pb.resetBasePositionAndOrientation(self._world_id, np.array([0., 0., -0.5]), np.array([0.,0.,0.,1]))

        self._robot    = BulletRobot(robot_id=robot_id, ee_link_idx=2, config=config_pih_world, enable_force_torque_sensors = True)

        self._peg.configure_default_pos(np.array([0, -1.1, 1.5]), np.array([0., 0., 0., 1]))

        self._robot.configure_default_pos(np.array([0, -1.3, 3.]),  np.array([0., 1, 0., 0]))

        self._config   = config

        self._record_sample = RecordSample(robot_interface=self._robot, 
                                          task_interface=self._peg,
                                          data_folder_path=self._config['data_folder_path'],
                                          data_name_prefix='sim_pih_data',
                                          num_samples_per_file=500)

    def step(self):

        pb.stepSimulation()

    def on_shutdown(self):

        #this if for saving files in case keyboard interrupt happens
        self._record_sample.save_data_now()

    def get_force_torque_details(self):

        ee_in_contact_with_box = False
        in_contact = False

        contact_point = self._robot.get_contact_points()

        if len(contact_point) > 4:

            in_contact = True

            if contact_point[2] == self._peg._id and contact_point[3] == self._robot._ee_link_idx:

                ee_in_contact_with_box = True

        if in_contact:

            [fx,fy,fz], [tx,ty,tz] = self._robot.get_joint_details(joint_idx = self._robot._ee_link_idx, flag = 'force_torque')

            if ee_in_contact_with_box:

                print "\n\nForce: ", fx, fy, fz
                print "\nTorque: ", tx, ty, tz

            else:
                print "Robot in contact with other object. Object ID:", "Square_Hole_Table" if contact_point[2] == self._hole._id else contact_point[2]
                print "Force: ", fx, fy, fz
                print "Torque: ", tx, ty, tz
        
    def run(self):

        self.rate = rospy.Rate(100)

        pb.setRealTimeSimulation(0)

        import time

        time.sleep(1)

        rospy.on_shutdown(self.on_shutdown)

        while not rospy.is_shutdown():

            self.get_force_torque_details()

            self.step()

            self.rate.sleep()

        pb.setRealTimeSimulation(1)



