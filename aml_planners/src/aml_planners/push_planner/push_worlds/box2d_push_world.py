from plotter import Plotter

import pygame
import numpy as np

from convert_tools import image2string
from Box2D import (b2Filter,)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

from box import Box
from finger import Finger

import copy
import os


class Box2DPushWorld(object):

    def __init__(self, config):

        self._config = config

        self._last_push = [[0,0,0,0]] #px,py,F,theta
        self._last_theta = 0

        self._clear_colour = (255,255,255)

        self._ppm = self._config['pixels_per_meter']
        self._dt = self._config['dt']

        self._clear_screen_enabled = True

        # --- pybox2d world setup ---
        # Create the world
        self._world = world(gravity=(0, 0), doSleep=True)

        self._box = Box(self._world, self._config['box_config'])

        self._num_fingers = self._config['num_fins']

        self._fingers = [Finger(self._world, self._config['fin_config'])  for _ in range(self._num_fingers)]

        self._objects = [self._box]

        for finger in self._fingers:
            self._objects.append(finger)

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


    def update(self, push_action):

        if len(push_action) != self._num_fingers:
            raise Exception("The number of fingers does not match push actions")

        for f in range(self._num_fingers):

            self._fingers[f].set_state((push_action[f][1], push_action[f][2], 0, 0, 0, 0))
            self._fingers[f].apply_push2(0., 0., push_action[f][3], push_action[f][4])
            #this is the push position in the world frame
            self._last_push.append([push_action[f][5], push_action[f][6], push_action[f][3], push_action[f][4]])


    def step(self):

        self._world.Step(self._dt, 1, 1)

    def get_frame(self, cam_pos = (0,0)):

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


    def draw(self, screen, cam_pos = (0,0)):

        curr_frame = self.get_frame(cam_pos)

        # p = self._box.get_image_point(self._last_push[:2])

        # push_dir = np.array([np.cos(self._last_theta),np.sin(self._last_theta)])

        # pi = self._last_push[:2] - push_dir*2
        # pi = self._box.get_image_point2(pi)

        # pi = (int(pi[0]),int(pi[1]))
        # p = (int(p[0]),int(p[1]))

        # pygame.draw.circle(curr_frame, (255,0,0), p, 10, 0)
        # pygame.draw.line(curr_frame, (0,200,50), pi, p, 5)
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

    def reset(self):

        self._box.reset()

        self._world.ClearForces()

    def get_state(self):

        state = self._box.get_state()

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

    def sample_push_action(self, alpha=None):

        if alpha is None:
            alpha = self.get_push_action()
        action = alpha
        # alpha = np.random.rand()
        # idx_0 = np.random.choice(len(vertices))
        # idx_1 = (idx_0+1)%len(vertices)

        # p0 = np.array(vertices[idx_0])
        # p1 = np.array(vertices[idx_1])

        vertices = self._box.get_vertices_local()

        if  0. <= alpha <= 0.25:
            alpha = alpha/0.25
            p0 = np.array(vertices[0])  
            p1 = np.array(vertices[1])
        elif 0.25 <= alpha <= 0.5:
            alpha = (alpha-0.25)/0.25
            p0 = np.array(vertices[1])  
            p1 = np.array(vertices[2])
        elif 0.5 <= alpha <= 0.75:
            alpha = (alpha-0.5)/0.25
            p0 = np.array(vertices[2])  
            p1 = np.array(vertices[3])
        elif 0.75 <= alpha <= 1.:
            alpha = (alpha-0.75)/0.25
            p0 = np.array(vertices[3])  
            p1 = np.array(vertices[0])
        else:
            print "Received alpha is \t", alpha
            print "This is unacceptable alpha, acceptable range is (0,1) please check the code."

        box_w, box_h = self._config['box_config']['box_dim']

        # self._box._dyn_body.angle = np.pi/6

        edge_point = p0*(1 - alpha) + alpha*p1#np.multiply(np.random.rand(2),[2*box_w,2*box_h]) - np.array([box_w,box_h])

        centre = np.zeros(2)#self._box.get_point((0,0))

        edge_vec = (p1 - p0)
        edge_vec /= np.linalg.norm(edge_vec)

        centre_vec = centre - edge_point
        centre_vec /= np.linalg.norm(centre_vec)

        push_vec = centre_vec - np.dot(centre_vec,edge_vec)*edge_vec 

        theta_offset = np.pi#2*np.random.rand()*(np.pi/6.0) - (np.pi/6.0)

        theta = np.fmod((np.arctan2(push_vec[1],push_vec[0])) + theta_offset,2*np.pi)

        self._last_theta = theta

        # if theta*180/np.pi > self._max_ang:
        #     self._max_ang = theta*180/np.pi
        # elif theta*180/np.pi < self._min_ang:
        #     self._min_ang = theta*180/np.pi

        # if theta_offset*180/np.pi > self._max_ang:
        #     self._max_ang = theta_offset*180/np.pi
        # elif theta_offset*180/np.pi < self._min_ang:
        #     self._min_ang = theta_offset*180/np.pi

        # if theta < 0:
        # print "Min Ang: ", self._min_ang, " Max Ang: ", self._max_ang

        f_mag = 2.# np.random.uniform(0, self._config['push_mag'])

        px, py = edge_point

        fx = f_mag*np.cos(theta)
        fy = f_mag*np.sin(theta)

        # print self._push_counter
        return [action, theta, f_mag, px, py, fx, fy]

    def sample_push_action2(self, alpha=None):

        if alpha is None:
            alpha = self.get_push_action()

        if isinstance(alpha, np.ndarray):
            alpha = float(alpha[0])
        
        action = alpha
        # alpha = np.random.rand()
        # idx_0 = np.random.choice(len(vertices))
        # idx_1 = (idx_0+1)%len(vertices)

        # p0 = np.array(vertices[idx_0])
        # p1 = np.array(vertices[idx_1])

        vertices = self._box.get_vertices_local()

        #modified for baxter box2d integration

        # if 0. <= alpha <= 0.25:
        #     alpha = (alpha)/0.25
        #     p0 = np.array(vertices[1])  
        #     p1 = np.array(vertices[2])
        # elif 0.25 < alpha <= 0.5:
        #     alpha = (alpha-0.25)/0.25
        #     p0 = np.array(vertices[3])  
        #     p1 = np.array(vertices[0])
        # elif  0.5 < alpha <= 0.75:
        #     alpha = (alpha-0.5)/0.25
        #     p0 = np.array(vertices[0])  
        #     p1 = np.array(vertices[1])
        # elif 0.75 < alpha <= 1.:
        #     alpha = (alpha-0.75)/0.25
        #     p0 = np.array(vertices[2])  
        #     p1 = np.array(vertices[3])
        # else:
        #     print "Received alpha is \t", alpha
        #     print "This is unacceptable alpha, acceptable range is (0,1) please check the code."

        if  0. <= alpha <= 0.25:
            alpha = alpha/0.25
            p0 = np.array(vertices[0])  
            p1 = np.array(vertices[1])
        elif 0.25 <= alpha <= 0.5:
            alpha = (alpha-0.25)/0.25
            p0 = np.array(vertices[1])  
            p1 = np.array(vertices[2])
        elif 0.5 <= alpha <= 0.75:
            alpha = (alpha-0.5)/0.25
            p0 = np.array(vertices[2])  
            p1 = np.array(vertices[3])
        elif 0.75 <= alpha <= 1.:
            alpha = (alpha-0.75)/0.25
            p0 = np.array(vertices[3])  
            p1 = np.array(vertices[0])
        else:
            print "Received alpha is \t", alpha
            print "This is unacceptable alpha, acceptable range is (0,1) please check the code."

        box_w, box_h = self._config['box_config']['box_dim']

        # self._box._dyn_body.angle = np.pi/6

        edge_point = p0*(1 - alpha) + alpha*p1#np.multiply(np.random.rand(2),[2*box_w,2*box_h]) - np.array([box_w,box_h])

        centre = np.zeros(2)#self._box.get_point((0,0))

        edge_vec = (p1 - p0)
        edge_vec /= np.linalg.norm(edge_vec)

        centre_vec = centre - edge_point
        centre_vec /= np.linalg.norm(centre_vec)

        push_vec = centre_vec - np.dot(centre_vec,edge_vec)*edge_vec 

        theta_offset = np.pi#2*np.random.rand()*(np.pi/6.0) - (np.pi/6.0)

        theta = np.fmod((np.arctan2(push_vec[1],push_vec[0])) + theta_offset,2*np.pi)

        self._last_theta = theta

        # if theta*180/np.pi > self._max_ang:
        #     self._max_ang = theta*180/np.pi
        # elif theta*180/np.pi < self._min_ang:
        #     self._min_ang = theta*180/np.pi

        # if theta_offset*180/np.pi > self._max_ang:
        #     self._max_ang = theta_offset*180/np.pi
        # elif theta_offset*180/np.pi < self._min_ang:
        #     self._min_ang = theta_offset*180/np.pi

        # if theta < 0:
        # print "Min Ang: ", self._min_ang, " Max Ang: ", self._max_ang

        f_mag = self._config['push_mag']# np.random.uniform(0, self._config['push_mag'])

        px, py = edge_point

        pre_push_pos_box_local = edge_point - push_vec*self._config['pre_push_offset']

        
        pre_push_pos_world = self._box.get_point(pre_push_pos_box_local)
        push_pos_world     = self._box.get_point(edge_point)

        fin_push_dir = (push_pos_world - pre_push_pos_world)/np.linalg.norm(pre_push_pos_world - push_pos_world)

        fx = f_mag*fin_push_dir[0]
        fy = f_mag*fin_push_dir[1]

        # print self._push_counter
        return [action, pre_push_pos_world[0], pre_push_pos_world[1], fx, fy, push_pos_world[0], push_pos_world[1]]


    def sample_push_action3(self, alpha=np.array([])):
        push_actions = []

        if not alpha.size:
            alpha = [self.get_push_action() for _ in range(self._num_fingers)]
        else:
            alpha = [None for _ in range(self._num_fingers)]

        for f in range(self._num_fingers):
            push_actions.append(self.sample_push_action2(alpha[f]))

        return push_actions










