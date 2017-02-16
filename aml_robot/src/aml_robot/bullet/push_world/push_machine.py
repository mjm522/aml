import cv2
import time
import math
import rospy
import random
import numpy as np
import pybullet as pb

from aml_robot.bullet.bullet_robot import BulletRobot
from aml_data_collec_utils.record_sample import RecordSample
from aml_robot.bullet.push_world.config import config_push_world

class BoxObject(BulletRobot):

    def __init__(self, box_id):

        super(BoxObject, self).__init__(robot_id=box_id, config=config_push_world)

    def get_effect(self):
        p, q   = self.get_pos_ori()
        status = {}
        status['pos'] = p
        #all the files in package follows np.quaternion convention, that is
        # w,x,y,z while ros follows x,y,z,w convention
        status['ori'] = np.array([q[3],q[0],q[1],q[2]])

        return status

class PushMachine():

    def __init__(self, world_id, box_id, robot_id, config):

        self._box      = BoxObject(box_id=box_id)

        self._world_id = world_id

        pb.resetBasePositionAndOrientation(self._world_id, np.array([0., 0., -0.5]), np.array([0.,0.,0.,1]))

        self._robot    = BulletRobot(robot_id=robot_id, ee_link_idx=-1, config=config_push_world)

        self._box.configure_default_pos(np.array([0., 0., 0.]), np.array([0.,0.,0.,1]))

        self._robot.configure_default_pos(np.array([1.25, 0.,0.]),  np.array([0.,0.,0.,1]))

        self._config   = config

        self._fsm = ['pre_push', 'push']

        self._fsm_state = 0

        self._force_mag = self._config['push_force_magnitude']

        self._record_sample = RecordSample(robot_interface=self._robot, 
                                          task_interface=self._box,
                                          data_folder_path=self._config['data_folder_path'],
                                          data_name_prefix='sim_push_data',
                                          num_samples_per_file=500)

    def step(self):

        pb.stepSimulation()

    def on_shutdown(self):

        #this if for saving files in case keyboard interrupt happens
        self._record_sample.save_data_now()


    def set_pre_push_location(self):

        length_div2  = self._config['box_type']['length']/2
        
        breadth_div2 = self._config['box_type']['breadth']/2

        pre_push_offsets = self._config['pre_push_offsets']

        if random.uniform(0.,1.) > 0.5:
            weight = 1.
        else:
            weight = -1.

        if random.uniform(0.,1.) > 0.5:

            x_box = random.uniform(-length_div2, length_div2)   # w.r.t box frame
            pre_push_wrt_box = np.array([x_box, weight*pre_push_offsets[1], 0.])
            
        else:

            y_box = random.uniform(-breadth_div2, breadth_div2) # w.r.t box frame
            pre_push_wrt_box = np.array([weight*pre_push_offsets[0], y_box, 0.])

        box_pos, box_ori = self._box.get_pos_ori()

        rot = np.asarray(pb.getMatrixFromQuaternion(box_ori)).reshape(3,3)

        push_location_wrt_world =  box_pos + pre_push_wrt_box

        self._robot.set_pos_ori(push_location_wrt_world, box_ori)

        push_info = {'push_xy':pre_push_wrt_box}

        return push_info


    def push_box(self):

        box_pos, box_ori = self._box.get_pos_ori()

        rbt_pos, rbt_ori = self._robot.get_pos_ori()

        rot = np.asarray(pb.getMatrixFromQuaternion(box_ori)).reshape(3,3)

        force_dir = np.dot(rot.T,(box_pos-rbt_pos))

        force =  self._force_mag*force_dir/np.linalg.norm(force_dir)

        self._robot.apply_external_force(link_idx=-1, force=force, point=[0.,0.,0.], local=True)


    def check_within_limits(self):

        def check_limits(pos, limits):
            
            crossed_xlim =  True
            crossed_ylim =  True
            crossed_zlim =  True

            if  limits['x_lower'] <= pos[0] <= limits['x_upper']:

                crossed_xlim = False

            if limits['y_lower'] <= pos[1] <= limits['y_upper']:

                crossed_ylim = False

            if limits['z_lower'] <= pos[2] <= limits['z_upper']:

                crossed_zlim = False

            return crossed_xlim or crossed_ylim or crossed_zlim


        box_pos, box_ori = self._box.get_pos_ori()

        rbt_pos, rbt_ori = self._robot.get_pos_ori()

        work_space_limits = self._config['work_space_limits']

        # if check_limits(box_pos, work_space_limits):

        #     self._box.set_default_pos_ori()


        if check_limits(rbt_pos, work_space_limits):

            self._robot.set_default_pos_ori()

            self._box.set_default_pos_ori()

    
    def fsm(self):

        if self._fsm[self._fsm_state] == 'pre_push':

            print "Going to pre-push location ..."

            push_info = self.set_pre_push_location()

            self._record_sample.record_once(task_action=push_info)

            # raw_input()

            self._fsm_state = 1

        elif self._fsm[self._fsm_state] == 'push':

            # print "Gonna push ..."

            def check_contact_with_box():
                in_contact = False

                contact_point = self._robot.get_contact_points()

                if len(contact_point) > 4:

                    if contact_point[1] == self._box._id:

                        in_contact = True

                return in_contact


            success = False

            self.push_box()

            if check_contact_with_box():

                count = 0

                #try to be in contact with the box for some time.
                while count < 10:

                    self.push_box()
                    
                    self.step()

                    self.rate.sleep()

                    if check_contact_with_box():

                        count += 1
                        # print count

                    self.check_within_limits()

                self._fsm_state = 0

                success = True

                # self._box.set_default_pos_ori()

                self._robot.set_default_pos_ori()

                self._record_sample.record_once(task_action=None, task_status=success)

        self.check_within_limits()
    
    def run(self):

        self.rate = rospy.Rate(100)

        pb.setRealTimeSimulation(0)

        import time

        time.sleep(1)

        rospy.on_shutdown(self.on_shutdown)

        while not rospy.is_shutdown():

            self.fsm()

            self.step()

            self.rate.sleep()

        pb.setRealTimeSimulation(1)



