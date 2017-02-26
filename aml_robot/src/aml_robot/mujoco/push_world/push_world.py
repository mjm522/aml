import numpy as np
import random
import rospy

from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer

import matplotlib
matplotlib.use("Qt4Agg") # For using matplotlib with pygame
import matplotlib.pyplot as plt

import pickle

from config import config_push_world

from aml_robot.box2d.data_manager import DataManager

STATE = {
    'RESET': 0,
    'NEW_PUSH': 1,
    'SAVE_DATA': 2,
    'APPLY_PUSH': 3,
    'NOP': 4,
}

class PushWorld(object):

    def __init__(self, config):

        self._config = config_push_world
        self._ppm = self._config['pixels_per_meter']

        # Create a dynamic body using mujoco
    
        self._box = MujocoRobot(xml_path=self._config['model_name'])

        self._dynamic_body = self._box._model 

        self._dt = self._box._dt

        self._last_push = [0,0,0,0,0]

        self._change_after = 200
        self._push_counter = 0
        self._next_idx = 0

        self._current_state = STATE['RESET']
        
        self._data_manager = DataManager()
        self._new_sample = self._data_manager.create_sample()


    def step(self):
        self._box.step()
                
    def get_screen_point2(self,body,local_point):
        px, py = body.transform*local_point*self._ppm
        py = self._config['image_height'] - py
        p = (px,py)

        return p

    def get_point(self,body,local_point):
        p = body.transform*local_point

        return p

    def get_screen_point(self,world_point):
        px = world_point[0]*self._ppm
        py = self._config['image_height'] - world_point[1]*self._ppm 
        
        p = (px,py)

        return p


    def reset_box(self):

        # self._box.reset_model()
        self._dynamic_body.data.qpos = np.array([0., 0., 0., 1, 0, 0, 0])
        self._dynamic_body.data.qvel = np.zeros(6)
        self._dynamic_body.data.qacc = np.zeros(6)
        self._dynamic_body.data.qfrc_applied = np.zeros((self._dynamic_body.nv,1))


    def get_box_state(self, body, viewer = None):
        #body should of type MujocoRobot.model
        #viewer should of type MjViewer
        qpos    = body.data.qpos
        qvel    = body.data.qvel

        image_file = None
        if viewer is not None:
            image_file = "images/img%d.png"%(self._next_idx,)
            if self._config['record_training_data']:
                self.save_screen(viewer.get_image(), image_file)
            
            self._next_idx += 1

        # Current state of the box (position,linear and angular velocities, image_rgb)
        # Final state of the box 


        print qpos
        print qvel

        state = {
            'position': np.array([qpos[0], qpos[1], qpos[2]]),
            'angle': np.array([qpos[3],qpos[4],qpos[5],qpos[6]]),
            'linear_velocity': np.array([qvel[0], qvel[1], qvel[2]]),
            'angular_velocity': np.array([qvel[3],qvel[4],qvel[5]]),
            'image_rgb': image_file,
        }

        return state


    def get_vertices(self):
        vertices = []

        for i in range(1,8):
            vertices.append(self._box.site_pose('c'+str(i))[0])

        # print vertices

        return vertices

    def generate_random_push(self):

        body = self._dynamic_body

        box_l, box_w, box_h = self._config['box_dim']

        px, py, pz = np.multiply(np.random.rand(3),[2*box_l, 2*box_w, 2*box_h]) - np.array([box_l, box_w, box_h])

        theta = np.random.rand()*np.pi*0.5#(2*np.pi)




        # self.get_vertices()

        self._push_counter += 1

        # print self._push_counter
        return 0.0, 0.0, 0.0, theta#px, py, theta

    def to_vec(self, theta):
        return np.cos(theta), np.sin(theta)


    def save_screen(self, img, filename):

        img = np.fromstring(img, dtype='uint8').reshape(self._config['image_height'], self._config['image_width'], 3)[::-1,:,:]
        
        if img is not None:
            matplotlib.image.imsave(filename, img)


    def add_sample(self, new_sample):

        new_sample['filled'] = True
        new_sample['stale'] = False

        if self._config['record_training_data']:

            self._data_manager.add(new_sample)


    def update(self, viewer):

        next_state = self._current_state

        body = self._dynamic_body

        if self._current_state == STATE['RESET']:
            px, py, pz, theta = self.generate_random_push()
            
            ix, iy = self.to_vec(theta); iz = 0;

            self._last_push = [px, py, pz, ix, iy, iz, theta]

            self._push_counter = 0

            self.reset_box()

            next_state = STATE['SAVE_DATA']
            self._new_sample = self._data_manager.create_sample()

        elif self._current_state == STATE['SAVE_DATA']:

            # Current state of the box (position,linear and angular velocities, image_rgb)
            # Final state of the box 
            # Current image
            # Last image

            if self._push_counter > 0:
                next_state = STATE['RESET']

                state = self.get_box_state(body, viewer)

                # self._new_sample['image_rgb_end'] = self._viewer._last_screen
                self._new_sample['state_end'] = state

                print "SAMPLE_ID:", self._data_manager._next_sample_id

                self.add_sample(self._new_sample)

            else:

                state = self.get_box_state(body, viewer)

                # self._new_sample['image_rgb_start'] = self._viewer._last_screen
                self._new_sample['state_start'] = state
                self._new_sample['push_action'] = np.array([self._last_push])

                next_state = STATE['APPLY_PUSH']

        elif self._current_state == STATE['APPLY_PUSH']:

            px, py, pz, ix, iy, iz, theta = self._last_push

            self.apply_push(body, px, py, pz, ix, iy, iz, theta)

            self._push_counter += 1

            if self._push_counter%self._change_after==0:

                next_state = STATE['SAVE_DATA']

        # print "CURRENT STATE:", self._current_state, " NEXT_STATE: ", next_state

        # Updating next state
        self._current_state = next_state

        return self._current_state 

    def apply_push(self, body, px, py, pz, ix, iy, iz, theta):

        # px_world, py_world = self.get_point(body,(px, py))
        
        ix, iy = self.to_vec(theta); iz = 0;

        ix *= 10; iy *= 10; #mass of the object is 1kg, so atleast this amount of force is needed

        # px = 0.; py = 0.; pz = 0.;
        # ix = 10.0;  iy = 0; iz = 0.0;

        print "px ", px, "py ", py, "pz ",pz
        print "ix ", ix, "iy ", iy, "iz ",iz

        qfrc_target = body.applyFT(point=np.array([px, py, pz]), 
                                   force=np.array([ix, iy, iz]), 
                                   torque=np.zeros(3), body_name='Box')

        print "qfrc target \t", np.round(qfrc_target.flatten(), 3)

        body.data.qfrc_applied = qfrc_target


    def save_samples(self,filename):

        self._data_manager.save(filename)

def main():
    
    push_world = PushWorld(config = config_push_world)

    viewer = MujocoViewer(mujoco_robot=push_world._box, width=config_push_world['image_width'], height=config_push_world['image_height'])

    viewer.configure(cam_pos=config_push_world['camera_pos'])

    while not rospy.is_shutdown():
        push_world.update(viewer=viewer)
        viewer.loop()

    push_world.save_samples(config_push_world['training_data_file'])

if __name__ == "__main__":

    rospy.init_node('mujoco_push_world')

    main()


