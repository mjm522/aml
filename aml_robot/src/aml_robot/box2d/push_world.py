import numpy as np
import pygame


import matplotlib
matplotlib.use("Qt4Agg") # For using matplotlib with pygame
import matplotlib.pyplot as plt

import threading

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

import pickle

from config import config
from pygame_viewer import PyGameViewer


from data_manager import DataManager

STATE = {
    'RESET': 0,
    'NEW_PUSH': 1,
    'SAVE_DATA': 2,
    'APPLY_PUSH': 3,
    'NOP': 4,
}


class PushWorld(object):

    def __init__(self,config):

        self._config = config
        self._ppm = self._config['pixels_per_meter']
        self._dt = self._config['dt']

        # --- pybox2d world setup ---
        # Create the world
        self._world = world(gravity=(0, 0), doSleep=True)

        # Create a dynamic body
        self._dynamic_body = self._world.CreateDynamicBody(position=(32, 24), angle=0)

        self._dynamic_body.linearDamping = 0.6
        self._dynamic_body.angularDamping = 0.05

        self._dynamic_body.awake = True

        # And add a box fixture onto it (with a nonzero density, so it will move)
        self._box = self._dynamic_body.CreatePolygonFixture(box=self._config['box_dim'], density=1, friction=0.3)

        self._colors = {
            staticBody: (255, 255, 255, 255),
            dynamicBody: (127, 127, 127, 255),
        }

        self._last_push = [0,0,0,0,0]

        self._change_after = 50
        self._push_counter = 0
        self._next_idx = 0

        self._current_state = STATE['RESET']
        
        self._data_manager = DataManager()
        self._new_sample = self._data_manager.create_sample()

    def step(self):
        self._world.Step(self._dt, 10, 10)

    def draw(self,viewer):

        screen = viewer._screen

        # Draw the world
        for body in (self._dynamic_body,):  # or: world.bodies
            
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
                # and also the shape.
                shape = fixture.shape

                # Naively assume that this is a polygon shape. (not good normally!)
                # We take the body's transform and multiply it with each
                # vertex, and then convert from meters to pixels with the scale
                # factor.
                vertices = [(body.transform * v) * self._ppm for v in shape.vertices]

                # Box2d uses cannonical axis, we need to convert to match the screen axis convention
                # (-y) 
                vertices = [(v[0], self._config['image_height'] - v[1]) for v in vertices]


                pygame.draw.polygon(screen, self._colors[body.type], vertices)

                px, py, ix, iy, theta = self._last_push
                ix, iy = self.to_vec(theta)

                p = self.get_screen_point2(body,(px,py))
                p = (int(p[0]),int(p[1]))
                
                pygame.draw.line(screen,(127,255,127,255),(p[0],p[1]),(int(p[0]+ix*70),int(p[1]-iy*70)), 6)
                pygame.draw.circle(screen, (127,255,127,255), p, 5,0)

                center = self.get_point(body,(0,0))

                vel = body.linearVelocity
                ang_vel = body.angularVelocity
                # render text
                label = viewer.create_text_surface("Push angle %0.2f, Pos (%0.2f,%0.2f), Vel(%0.2f,%0.2f) AngVel %0.2f"%(theta*180/np.pi,center[0],center[1],vel[0],vel[1],ang_vel))
                screen.blit(label, (0, 10))


    def handle_event(self, event):
        """ Implement if needed """
        pass
                
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
        body = self._dynamic_body

        body.position = (16, 12)
        body.angle = 0
        body.linearVelocity = (0,0)
        body.angularVelocity = 0

    def get_box_state(self,body, viewer = None):

        (px,py) = body.position
        angle = body.angle
        (vx,vy) = body.linearVelocity
        omega = body.angularVelocity

        image_file = None
        if viewer is not None:
            image_file = "images/img%d.png"%(self._next_idx,)
            if self._config['record_training_data']:
                self.save_screen(viewer._last_screen,image_file)
            
            self._next_idx += 1

        # Current state of the box (position,linear and angular velocities, image_rgb)
        # Final state of the box 

        state = {
            'position': np.array([px,py]),
            'angle': angle,
            'linear_velocity': np.array([vx,vy]),
            'angular_velocity': omega,
            'image_rgb': image_file,
        }

        return state


    def get_vertices(self):
        
        body = self._dynamic_body

        shape = body.fixtures[0].shape

        vertices = [(body.transform * v) for v in shape.vertices]

        # print vertices

        return vertices

    def generate_random_push(self):

        body = self._dynamic_body

        box_w, box_h = self._config['box_dim']

        px, py = np.multiply(np.random.rand(2),[2*box_w,2*box_h]) - np.array([box_w,box_h])

        theta = np.random.rand()*np.pi*2#*0.5#(2*np.pi)

        # self.get_vertices()



        self._push_counter += 1

        # print self._push_counter
        return 0.0, 0.0, theta

    def to_vec(self,theta):

        return np.cos(theta + self._dynamic_body.angle), np.sin(theta + self._dynamic_body.angle)


    def save_screen(self,img,filename):

        # img = pygame.surfarray.pixels3d(screen) 
        # img = img.transpose(1,0,2)
        # print "SHAPE:", img.shape
        # im = Image.fromarray(img)
        # im.save("your_file.jpeg")

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
            px, py, theta = self.generate_random_push()
            ix, iy = self.to_vec(theta)

            self._last_push = [px, py, ix, iy, theta]

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

                print "END: ", state['position'], state['angle']
                print "SAMPLE_ID:", self._data_manager._next_sample_id


                self.add_sample(self._new_sample)

            else:

                state = self.get_box_state(body, viewer)

                # self._new_sample['image_rgb_start'] = self._viewer._last_screen
                self._new_sample['state_start'] = state
                self._new_sample['push_action'] = np.array([self._last_push])

                print "START: ", state['position'], state['angle']

                next_state = STATE['APPLY_PUSH']

        elif self._current_state == STATE['APPLY_PUSH']:

            px, py, ix, iy, theta = self._last_push

            if self._push_counter < 2:
                self.apply_push(body, px, py, ix, iy, theta)

            self._push_counter += 1

            if self._push_counter%self._change_after==0:

                next_state = STATE['SAVE_DATA']

        # print "CURRENT STATE:", self._current_state, " NEXT_STATE: ", next_state

        # Updating next state
        self._current_state = next_state

        return self._current_state 

    def apply_push(self,body, px, py, ix, iy, theta):

        px_world, py_world = self.get_point(body,(px,py))
        ix, iy = self.to_vec(theta)

        body.ApplyLinearImpulse(impulse=(ix*10,iy*10), point=(px_world,py_world), wake=True)


    def save_samples(self,filename):

        self._data_manager.save(filename)


def main():
    
    push_world = PushWorld(config = config)

    viewer = PyGameViewer(push_world, config = config)

    viewer.loop()

    push_world.save_samples(config['training_data_file'])


if __name__ == "__main__":
    main()


