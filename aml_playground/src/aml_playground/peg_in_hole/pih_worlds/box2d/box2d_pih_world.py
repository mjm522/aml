import os
import copy
import pygame
import numpy as np
from box import Box
from plotter import Plotter
from Box2D import (b2Filter,)
from polygon import Polygon
from manipulator import Manipulator
from contact_sensor import ContactSensor
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)


class Box2DPIHWorld(object):

    def __init__(self, config):

        self._config = config

        self._last_push = [[0,0,0,0]] #px,py,F,theta
        self._last_theta = 0

        self._clear_colour = (255,255,255)

        self._ppm = self._config['pixels_per_meter']
        self._dt = self._config['dt']

        self._clear_screen_enabled = True

        self._config['man_config']['dt'] = self._dt

        # --- pybox2d world setup ---
        # Create the world
        self._world = world(gravity=(0, 0), doSleep=True, contactListener=ContactSensor())

        self._box = Box(self._world, self._config['box_config'])

        self._manipulator = Manipulator(self._world, self._config['man_config'])

        self._hole = Polygon(self._world, self._config['hole_config'])

        self._objects = [self._box, self._manipulator, self._hole]

        self._nu = self._manipulator._num_links

        self._surface = pygame.Surface((config['image_width'], config['image_height']))

        self._min_ang = 360
        self._max_ang = -360

        self._cost_history = []

        self._plotter = Plotter()

        self._goal = None

        self._obstacle = None


    def update_goal_location(self, goal):
        #expects a list or array of the following format[x, y, th]
        if goal:
            self._goal = goal
        else:
            self._goal = None


    def update_obs_location(self, obstacle):
        #expects a list [x,y, radius]
        if obstacle:
            self._obstacle =  obstacle
        else:
            self._obstacle = None

    def cost_map(self):

        X = []; Y = []; Z = []

        if self._obstacle is not None:
            delta = 0.025
            x = np.arange(0, 32, delta)
            y = np.arange(0, 24, delta)
            obstacle = self._obstacle

            X, Y = np.meshgrid(x - obstacle[0], y - obstacle[1])
            Z = np.sqrt(X**2 + Y**2)
        
        return X, Y, Z


    def cost(self, x, u):

        if self._obstacle is not None:
            obstacle = self._obstacle[:2]
            distv = obstacle - x[:2]
            return np.exp(1./(np.dot(distv,distv) + 1e-5))
        else:
            return 0.


    def update(self, action):

        self._manipulator.set_joint_speed(action)


    def step(self):

        self._world.Step(self._dt, 1, 1)

    def get_frame(self, cam_pos=(0., 0.)):
        
        if self._clear_screen_enabled:
            self._surface.fill(self._clear_colour) # paint it white

        self._plotter.draw(self._surface)

        # Draw the world
        for obj in self._objects:
            obj.draw(self._surface, cam_pos)
        
        return self._surface


    def get_cv_frame(self):
        img = copy.deepcopy(pygame.surfarray.pixels3d(self.get_frame()))
        
        return copy.deepcopy(img.transpose(1,0,2))


    def draw(self, screen, cam_pos=(0., 0.)):

        curr_frame = self.get_frame(cam_pos)

        if self._goal is not None:
            goal_pixel = self._box.get_image_point(self._goal[:2])
            pygame.draw.rect(curr_frame, (0,255,0), (int(goal_pixel[0] - cam_pos[0]), int(goal_pixel[1] - cam_pos[1]), 20, 20))

        if self._obstacle is not None:
            obstacle_pixel = self._box.get_image_point(self._obstacle[:2])
            pygame.draw.circle(curr_frame, (255,0,0), (int(obstacle_pixel[0] - cam_pos[0]), int(obstacle_pixel[1] - cam_pos[1])), int(self._obstacle[2]*self._ppm),0)

        screen.blit(curr_frame, (0, 0))

    
    def handle_event(self, event):
        """ Implement if needed """

        (pressed1,pressed2,pressed3) = pygame.mouse.get_pressed()

        if pressed1==1:
            pos = np.array(pygame.mouse.get_pos())
            
            pos[0] /= self._ppm
            pos[1] = self._config['image_height'] - pos[1]
            pos[1] /= self._ppm

            print "POS: ", pos

            self._box._dyn_body.position = pos

            x = self.pack_box_state()


            cost = self.cost(x,self._last_push)

            self._cost_history.append(cost)

            print "Cost: ", cost

            # self._plotter.plot(range(len(self._cost_history)),self._cost_history)

        if event.type == pygame.MOUSEBUTTONDOWN:
            pass
            # X, Y, Z = self.cost_map()

            # self._plotter.contour(X,Y,Z)

    def reset(self, noise=0.5):

        self._box.reset(noise)

        self._manipulator.reset(noise)

        self._world.ClearForces()

    def get_state(self):

        state = {'box':self._box.get_state(),
                 'manipulator':self._manipulator.get_state()}
        
        # image_rgb = self.get_cv_frame()
        # image_rgb = image2string(image_rgb)

        # state['image_rgb'] = image_rgb

        return state

    def set_state(self, state):
        self._box.set_state(state)

    def pack_box_state(self):

        state = self._box.get_state()

        return np.r_[state['position'],state['angle'],state['linear_velocity'],state['angular_velocity']]

    def get_push_action(self):
        push_action_left  = np.random.uniform(0, 0.25) #me left when i face the robot
        push_action_right = np.random.uniform(0.25, 0.5)
        push_action_front = np.random.uniform(0.5, 0.75)
        push_action_back  = np.random.uniform(0.75, 1.)
         
        push_actions = [push_action_left, push_action_right, push_action_front, push_action_back]

        return push_actions[np.random.randint(0,4)] #np.random.randint(0,4)

    def sample_action(self, alpha=None):

        return self._manipulator.sample_action()

