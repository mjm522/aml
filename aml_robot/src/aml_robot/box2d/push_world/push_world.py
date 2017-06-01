import pygame
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg") # For using matplotlib with pygame
import matplotlib.pyplot as plt
from aml_io.convert_tools import image2string
from aml_robot.box2d.box2d_robot import Box2DRobot
from aml_robot.box2d.core.data_manager import DataManager

import copy
import os


STATE = {
    'RESET': 0,
    'NEW_PUSH': 1,
    'SAVE_DATA': 2,
    'APPLY_PUSH': 3,
    'NOP': 4,
}

DEBUG = False

class PushWorld(Box2DRobot):

    def __init__(self, config):

        Box2DRobot.__init__(self, config=config)

        self._config = config

        # Create a dynamic body
        self._dynamic_body = self._world.CreateDynamicBody(position=(32, 24), angle=0)

        self._dynamic_body.linearDamping = 0.6
        self._dynamic_body.angularDamping = 0.05

        self._dynamic_body.awake = True

        # And add a box fixture onto it (with a nonzero density, so it will move)
        self._box = self._dynamic_body.CreatePolygonFixture(box=self._config['box_dim'], density=1, friction=0.3)

        self._last_push = [0,0,0,0] #px,py,F,theta

        self._change_after = 50
        self._push_counter = 0
        self._next_idx = 0

        self._current_state = STATE['RESET']
        
        self._data_manager = DataManager(data_folder=config['data_folder_path'])
        self._new_sample = self._data_manager.create_sample()


        self._surface = pygame.Surface((config['image_width'], config['image_height']))

        print "Surface is None?", self._surface is None

    def step(self):
        self._world.Step(self._dt, 10, 10)

    def get_frame(self):

        self._surface.fill((0,0,0)) # paint it white

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

                pygame.draw.polygon(self._surface, self._colors[body.type], vertices)
        

        return self._surface


    def get_cv_frame(self):
        img = copy.deepcopy(pygame.surfarray.pixels3d(self.get_frame()))
        
        return copy.deepcopy(img.transpose(1,0,2))


    def draw(self, viewer, view_info=True):

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

                if view_info:
                    
                    px, py, f_mag, theta = self._last_push
                    ix, iy = self.to_vec(theta)

                    p = self.get_screen_point2(body,(px,py))
                    p = (int(p[0]),int(p[1]))
                    
                    pygame.draw.line(screen,(127,255,127,255),(p[0],p[1]),(int(p[0]+ix*70),int(p[1]-iy*70)), 6)
                    pygame.draw.circle(screen, (127,255,127,255), p, 5,0)

                    center = self.get_point(body,(0,0))

                    # render text
                    label = viewer.create_text_surface("Push angle %0.2f, Pos (%0.2f,%0.2f), Vel(%0.2f,%0.2f) AngVel %0.2f"%(theta*180/np.pi,center[0],center[1],vel[0],vel[1],ang_vel))
                    screen.blit(label, (0, 10))

                    vel = body.linearVelocity
                    ang_vel = body.angularVelocity


    def handle_event(self, event):
        """ Implement if needed """
        pass

    def reset_box(self):
        body = self._dynamic_body

        body.position = self._config['box_pos'] + np.random.randn(2)*np.sqrt(0.5)
        body.angle = self._config['box_ori']
        body.linearVelocity = (0,0)
        body.angularVelocity = 0

    def get_box_state(self, body, viewer = None):

        (px,py) = body.position
        angle = body.angle
        (vx,vy) = body.linearVelocity
        omega = body.angularVelocity

        image_file = None
        # if viewer is not None:
            # viewer.store_screen()
        if self._config['record_training_data']:
            image_file = self.get_cv_frame()
            if image_file is not None:
                image_file = image2string(image_file)
                # image_file = self._config['data_folder_path']+"/img%d.png"%(self._next_idx,) 
                # self.save_screen(viewer._last_screen, image_file)
                
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

        f_mag =  10.

        # self.get_vertices()

        self._push_counter += 1

        # print self._push_counter
        return 0.0, 0.0, f_mag, theta

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


    def update(self, viewer = None):

        next_state = self._current_state

        body = self._dynamic_body

        if self._current_state == STATE['RESET']:
            px, py, f_mag, theta = self.generate_random_push()

            self._last_push = [px, py, f_mag, theta]

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

                if DEBUG:
                    if not os.path.exists('tmp/'):
                        os.makedirs('tmp/')

                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    # matplotlib.image.imsave("tmp/After%d.jpg"%(self._new_sample['sample_id'],), self.get_cv_frame())

            else:

                state = self.get_box_state(body, viewer)

                # self._new_sample['image_rgb_start'] = self._viewer._last_screen
                self._new_sample['state_start'] = state
                self._new_sample['push_action'] = np.array([self._last_push])

                if DEBUG:
                    if not os.path.exists('tmp/'):
                        os.makedirs('tmp/')

                    matplotlib.image.imsave("tmp/Before%d.jpg"%(self._data_manager._next_sample_id,), self.get_cv_frame())


                print "START: ", state['position'], state['angle']

                next_state = STATE['APPLY_PUSH']

        elif self._current_state == STATE['APPLY_PUSH']:

            px, py, f_mag, theta = self._last_push

            if self._push_counter < 2:
                self.apply_push(body=body, px=px, py=py, force_mag=10., theta=theta)

            self._push_counter += 1

            if self._push_counter%self._change_after==0:

                next_state = STATE['SAVE_DATA']

        # print "CURRENT STATE:", self._current_state, " NEXT_STATE: ", next_state
        # Updating next state
        self._current_state = next_state

        return self._current_state 

    def apply_push(self, body, px, py, force_mag=10., theta=0.):

        self.apply_external_force(body=body, force_mag=force_mag, theta=theta, point=(px,py))

        px_world, py_world = self.get_point(body,(px,py))
        ix, iy = self.to_vec(theta)

        body.ApplyLinearImpulse(impulse=(ix*force_mag,iy*force_mag), 
                                point=(px_world,py_world), wake=True)


    def save_samples(self, filename):

        self._data_manager.save(self._config['data_folder_path']+'/'+filename)


    def to_vec(self, theta):

        return np.cos(theta + self._dynamic_body.angle), np.sin(theta + self._dynamic_body.angle)
