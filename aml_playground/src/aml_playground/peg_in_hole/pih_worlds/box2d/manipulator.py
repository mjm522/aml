import pygame
import numpy as np
from aml_playground.peg_in_hole.utilities.pykdl_box2d import PyKDLBox2d
from Box2D import (b2_pi, b2Filter, b2FixtureDef, b2CircleShape, b2PolygonShape)
from Box2D import (b2ContactListener, b2DestructionListener, b2DrawExtended)


class Manipulator(PyKDLBox2d):

    def __init__(self, world, config):

        super(Manipulator, self).__init__(config)

        self._config = config

        self._dt    = self._config['dt']

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

        if len(params['links']) - len(params['joints']) != 1:
            raise Exception("There are not sufficient number of joints!")

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

    def compute_os_ctrlr_cmd(self, os_set_point, Kp=0.1):
        state = self.get_state()
        error = Kp*np.dot(np.linalg.pinv(state['ee_jac'], rcond=1e-4), (os_set_point - state['ee_pos'])) - np.sqrt(Kp)*state['j_vel']

        return error

    def set_joint_pos(self, joint_pos):
        for k in range(1, len(self._bodies)):
            body = self._bodies[k]
            body.angle = joint_pos[k-1]


    def set_max_joit_torque(self, joint_torques):

        for k in range(len(self._joints)):
            self._joints[k].maxMotorTorque = joint_torques[k]

    # def set_joint_position(self, joint_pos):
    #     #this is on the assumption that joint frames and body frames are alligned
    #     #the first body = base is a static body
    #     for k in range(1,len(self._bodies)):
    #         self._bodies[k].angle = joint_pos[k]
    
    def get_vertices_phys(self, body):

        vertices = [(self._dyn_body.transform * v) for v in body.shape.vertices]

        return vertices

    def get_vertices_local(self, body):

        vertices = body.shape.vertices

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

        ee_pos = self.get_state()['ee_pos']
        
        for k in range(len(self._links)):    
            vertices = self.get_vertices(link_idx=k, cam_pos=cam_pos)
            pygame.draw.polygon(surface, self._link_color[k], vertices)

        pygame.draw.circle(surface, (0,0,0), (int(ee_pos[0] * self._ppm - cam_pos[0]),int( self._config['image_height'] - ee_pos[1] * self._ppm - cam_pos[1])), 10, 0)


    def get_state(self):

        link_pos = []
        link_vel = []
 
        joint_state = [joint.angle for joint in self._joints]
        joint_velocity = [joint.speed for joint in self._joints]
        joint_torques = [joint.GetMotorTorque(self._dt) for joint in self._joints]

        self.update_chain(joint_state)

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
                  'ee_pos': self.compute_fwd_kinematics(),
                  'ee_jac': self.compute_jacobian(),
                }

        return state


    def sample_action(self):
        state = self.get_state()
        joint_speed = state['j_vel'] + 0.9*np.ones(len(self._joints))
        return joint_speed


    def get_image_point(self,world_point):
        px = world_point[0]*self._ppm
        py = self._config['image_height'] - world_point[1]*self._ppm 

        return np.array([px,py])


    def get_image_point_norm(self,local_point):
        (px,py) = self.get_image_point2(local_point)

        px /= float(self._config['image_width'])
        py /= float(self._config['image_height'])

        return np.array([px,py])


    def reset(self, noise = 0.5):

        for body, link in zip(self._bodies, self._config['links']):
            #incase we have to add noise into the manipulator position
            #if not just set it to the default state

            if link['type'] == 'static':
                #the first body is a static body since it is the base
                #hence first link corresponds to that,
                #so we skip that and start with the next set of
                #movable links
                continue

            #add noise if we want to
            body_pos_noise = np.random.randn(2)*np.sqrt(noise)
            body_ori_noise = np.random.randn(1)*np.sqrt(noise)

            body.position = (link['pos'][0]+body_pos_noise[0], link['pos'][1]+body_pos_noise[1])
            body.angle    = link['ori'] + body_ori_noise[0]
            body.linearVelocity  = (0., 0.)
            body.angularVelocity = 0.


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
