import pygame
from Box2D import (b2Filter,)
import numpy as np


class Box(object):

    def __init__(self, world, config):

        self._config = config

        self._world = world

        self._ppm = self._config['pixels_per_meter']

        self.add_box(config)


    def add_box(self, params):
        # Create a dynamic body
        self._dyn_body = self._world.CreateDynamicBody(position=params['pos'], 
                                                       angle=params['ori'],
                                                       linearDamping=params['lin_damp'],
                                                       angularDamping=params['ang_damp'],
                                                       awake=params['awake'])
        
        self._box = self._dyn_body.CreatePolygonFixture(box=params['dim'], 
                                                        density=params['den'], 
                                                        friction=params['mu'])
        self._colour = (125,125,125)

    def get_vertices_phys(self):

        vertices = [(self._dyn_body.transform * v) for v in self._box.shape.vertices]


        return vertices

    def get_vertices_local(self):

        vertices = self._box.shape.vertices

        return vertices

    def get_vertices(self, cam_pos = (0,0)):

        # The fixture holds information like density and friction,
        # and also the shape.
        shape = self._box.shape

        # Naively assume that this is a polygon shape. (not good normally!)
        # We take the body's transform and multiply it with each
        # vertex, and then convert from meters to pixels with the scale
        # factor.
        vertices = [(self._dyn_body.transform * v) * self._ppm for v in shape.vertices]

        # Box2d uses cannonical axis, we need to convert to match the screen axis convention
        # (-y) 
        vertices = [(v[0] - cam_pos[0], self._config['image_height'] - v[1] - cam_pos[1]) for v in vertices]


        return vertices

    def draw(self, surface, cam_pos = (0,0)):

        vertices = self.get_vertices(cam_pos)

        pygame.draw.polygon(surface, self._colour, vertices)

        cs = [(125,0,0), (0,125,0), (0,0,125), (125,125,125)]

        if self._config['draw_corners']:
            for idx in range(len(vertices)):
                v = vertices[idx]
                c = cs[idx]
                pygame.draw.circle(surface, c, (int(v[0]),int(v[1])), 5, 0)
            


    def get_angle(self):

        return self._dyn_body.angle



    def get_state(self):

        (px,py) = self._dyn_body.position
        angle = self._dyn_body.angle
        (vx,vy) = self._dyn_body.linearVelocity
        omega = self._dyn_body.angularVelocity

        state = { 'position': np.array([px,py]), 
                  'angle': angle, 
                  'linear_velocity': np.array([vx,vy]),
                   'angular_velocity': omega
                }

        return state


    def to_vec_world(self, theta):

        return np.array([np.cos(theta + self._dyn_body.angle), np.sin(theta + self._dyn_body.angle)])


    def get_point(self,local_point):
        p = self._dyn_body.transform*local_point

        return np.array([p[0],p[1]])

    def get_image_point(self,world_point):
        px = world_point[0]*self._ppm
        py = self._config['image_height'] - world_point[1]*self._ppm 


        return np.array([px,py])

    def get_image_point2(self,local_point):
        px, py = self._dyn_body.transform*local_point*self._ppm
        py = self._config['image_height'] - py

        return np.array([px,py])

    def get_image_point_norm(self,local_point):
        (px,py) = self.get_image_point2(local_point)

        px /= float(self._config['image_width'])
        py /= float(self._config['image_height'])

        return np.array([px,py])


    def apply_push(self, px, py, force_mag=10., theta=0.):

        px_world, py_world = self.get_point((px,py))
        ix, iy = self.to_vec_world(theta)

        self._dyn_body.ApplyLinearImpulse(impulse=(ix*force_mag,iy*force_mag), 
                                point=(px_world,py_world), wake=True)

    def apply_push2(self, px, py, fx, fy):

        px_world, py_world = self.get_point((px,py))

        print "px_world py_world", px_world, py_world

        self._dyn_body.ApplyLinearImpulse(impulse=(fx,fy), 
                                point=(px_world,py_world), wake=True)


    def apply_push3(self, px, py, fx, fy):

        px_world, py_world = self.get_point((px,py))

        self._dyn_body.ApplyForce(force=(fx,fy), 
                                point=(px_world,py_world), wake=True)   

    def reset(self, noise = 0.5):
        body = self._dyn_body

        body.position = self._config['pos'] + np.random.randn(2)*np.sqrt(noise)
        body.angle = self._config['ori']
        body.linearVelocity = (0,0)
        body.angularVelocity = 0

    def set_state(self, state):
        '''
        assumes following state order
        state = [x, y, th, dx, dy, dth]
        '''
        body = self._dyn_body
        body.position = (state[0], state[1])
        body.angle = state[2]
        body.linearVelocity = (state[3],state[4])
        body.angularVelocity = state[5]
