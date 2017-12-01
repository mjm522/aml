import cv2
import math
import numpy as np
import pandas as pd
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

    def __init__(self, world_id, peg_id, hole_id, robot_id, config, gains = [1, 1, 1]):

        self._peg      = BoxObject(box_id=peg_id)

        self._world_id = world_id

        self._hole = PegHole(hole_id, [0,0,1], [0, 0, -0.707, 0.707])

        pb.resetBasePositionAndOrientation(self._world_id, np.array([0., 0., -0.5]), np.array([0.,0.,0.,1]))

        self._robot    = BulletRobot(robot_id=robot_id, ee_link_idx=2, config=config_pih_world, enable_force_torque_sensors = True)

        self._peg.configure_default_pos(np.array([0, -1, 1.5]), np.array([0., 0., 0., 1]))

        self._robot.configure_default_pos(np.array([0, -1.3, 3.]),  np.array([0., 1, 0., 0]))

        self._config   = config

        self._gains = gains
        
        self._forces = []
        self._torques = []
        self._contact_points = []

    def step(self):

        pb.stepSimulation()

    def on_shutdown(self):

        d = {'contact point' : pd.Series(self._contact_points),
             'forces'        : pd.Series(self._forces),
             'torques'       : pd.Series(self._torques)}

        self._forces = []; self._torques = []; self._contact_points = []

        df = pd.DataFrame(d)
        df = df.rename_axis('Gains: '+str(self._gains), axis=1)
        file_name = self._config['data_folder_path']+'pih '+str(self._gains)+'.csv'
        df.to_csv(file_name)
        print df['contact point']

    def get_force_torque_details(self):

        ee_in_contact_with_box = False
        in_contact = False

        [fx,fy,fz,tx,ty,tz] = [0,0,0,0,0,0]
        contact_point = [0.,0.,0.]

        contact_details = self._robot.get_contact_points()

        if len(contact_details) > 4:

            in_contact = True

            if contact_details[2] == self._peg._id and contact_details[3] == self._robot._ee_link_idx:

                ee_in_contact_with_box = True

        if in_contact:

            [fx,fy,fz], [tx,ty,tz] = self._robot.get_joint_details(joint_idx = self._robot._ee_link_idx, flag = 'force_torque')

            if ee_in_contact_with_box:

                # print "\n\nForce: ", fx, fy, fz
                # print "\nTorque: ", tx, ty, tz
                contact_point = contact_details[5]

            else:
                print "Robot in contact with other object. Object ID:", "Square_Hole_Table" if contact_details[2] == self._hole._id else contact_details[2]
                print "Force: ", fx, fy, fz
                print "Torque: ", tx, ty, tz

        self._forces.append((fx,fy,fz))
        self._torques.append((tx,ty,tz))
        self._contact_points.append(contact_point)

        # print len(self._forces), len(self._torques), len(self._contact_points)
        
    def run(self):

        self.rate = rospy.Rate(100)

        pb.setRealTimeSimulation(0)

        import time

        time.sleep(1)

        rospy.on_shutdown(self.on_shutdown)

        # self._record_sample.start_record(task_action=pushes[idx])

        while not rospy.is_shutdown():

            self.get_force_torque_details()

            self.step()

            self.rate.sleep()

        pb.setRealTimeSimulation(1)



