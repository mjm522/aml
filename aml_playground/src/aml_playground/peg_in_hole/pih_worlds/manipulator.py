import pygame
import numpy as np
from Box2D import (b2_pi, b2Filter, b2FixtureDef, b2CircleShape, b2PolygonShape)
from Box2D import (b2ContactListener, b2DestructionListener, b2DrawExtended)



class Manipulator(object):

    def __init__(self, world, config):

        self._config = config

        self._world = world

        self._ppm = self._config['pixels_per_meter']

        self._num_links = len(self._config['links'])
        self._bodies = []
        self._links = []
        self._link_color = []
        self._joints = []

        self.add_manipulator(config)


    def add_manipulator(self, params):

        for link in params['links']:
            self._link_color.append(link['color'])
            
            if link['type'] ==  'static':
                body = self._world.CreateStaticBody(position=link['pos'])

            elif link['type'] == 'dynamic':
                body = self._world.CreateDynamicBody(position=link['pos'], 
                                                     angle=link['ori'],
                                                     linearDamping=link['lin_damp'],
                                                     angularDamping=link['ang_damp'],
                                                     awake=link['awake'])
            else:
                raise Exception("Unknown type")


            shape = body.CreatePolygonFixture(box=link['dim'], 
                                  density=link['den'], 
                                  friction=link['mu'])
            
            self._bodies.append(body)
            self._links.append(shape)

        k = 0
        for joint in params['joints']:
            joint = self._world.CreateRevoluteJoint(
                                    bodyA          = self._bodies[k], 
                                    bodyB          = self._bodies[k+1], 
                                    anchor         = joint['anchor'],
                                    lowerAngle     = joint['lowerAngle'],
                                    upperAngle     = joint['upperAngle'],
                                    enableLimit    = joint['enableLimit'],
                                    maxMotorTorque = joint['maxMotorTorque'],
                                    motorSpeed     = joint['motorSpeed'],
                                    enableMotor    = joint['enableMotor'],
                                    )
            self._joints.append(joint)
            k += 1


    def set_joint_speed(self, joint_speed):
        for k in range(len(self._joints)):
            self._joints[k].motorSpeed = joint_speed[k]
    
    def get_vertices_phys(self):

        vertices = [(self._dyn_body.transform * v) for v in self._fin.shape.vertices]

        return vertices

    def get_vertices_local(self):

        vertices = self._fin.shape.vertices

        return vertices

    def get_vertices(self, link_idx, cam_pos = (0,0)):

        # The fixture holds information like density and friction,
        # and also the shape

        shape = self._links[link_idx].shape

        # Naively assume that this is a polygon shape. (not good normally!)
        # We take the body's transform and multiply it with each
        # vertex, and then convert from meters to pixels with the scale
        # factor.
        vertices = [(self._bodies[link_idx].transform * v) * self._ppm for v in shape.vertices]

        # Box2d uses cannonical axis, we need to convert to match the screen axis convention
        # (-y) 
        vertices = [(v[0] - cam_pos[0], self._config['image_height'] - v[1] - cam_pos[1]) for v in vertices]


        return vertices

    def draw(self, surface, cam_pos = (0,0)):
        
        for k in range(len(self._links)):    
            vertices = self.get_vertices(link_idx=k, cam_pos=cam_pos)
            pygame.draw.polygon(surface, self._link_color[k], vertices)


    def get_angle(self):

        return self._dyn_body.angle



    def get_state(self):

        link_pos = []
        link_vel = []
 
        joint_state = [joint.angle for joint in self._joints]
        joint_velocity = [joint.speed for joint in self._joints]
        joint_torques = [joint.GetMotorTorque for joint in self._joints]

        #assuming first link to be base link
        for k in range(1,len(self._bodies)):
            (px,py) = self._bodies[k].position
            angle   = self._bodies[k].angle
            (vx,vy) = self._bodies[k].linearVelocity
            omg     = self._bodies[k].angularVelocity

            link_pos.append([px, py, angle])
            link_vel.append([vx, vy, omg])

        state = { 'j_pos': np.asarray(joint_state), 
                  'j_vel': np.asarray(joint_velocity), 
                  'j_torq':np.asarray(joint_torques),
                  'link_pos':np.asarray(link_pos),
                  'link_vel': np.asarray(link_vel),
                }

        return state


    def sample_action(self):
        state = self.get_state()
        joint_speed = state['j_vel'] + 0.9*np.ones(len(self._joints))
        return joint_speed


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
        for k in range(1, len(self._bodies)):
            body = self._bodies[k]
            body.position = (state[k+0], state[k+1])
            body.angle = state[k+2]
            body.linearVelocity = (state[k+3],state[k+4])
            body.angularVelocity = state[k+5]
