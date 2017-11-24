import pygame
from Box2D import (b2Filter,)
import numpy as np


class Polygon(object):

    def __init__(self, world, config):

        self._config = config

        self._world = world

        self._ppm = self._config['pixels_per_meter']

        self._colour = self._config['color']

        if self._config['type'] == 'dynamic':
            # Create a dynamic body
            self._body = self._world.CreateDynamicBody(position=config['pos'], 
                                                           angle=config['ori'], 
                                                           fixedRotation=False,
                                                           linearDamping=0.,
                                                           angularDamping=0.,
                                                           inertia=config['inertia'],
                                                           mass=config['mass'],
                                                           awake=True)


            

            # And add a box fixture onto it (with a nonzero density, so it will move)
            self._body_shape = self._body.CreatePolygonFixture(vertices=config['vertices'], 
                                                               density=1, 
                                                               friction=0.3, 
                                                               filter=b2Filter(groupIndex=-8,))
        elif self._config['type'] == 'static':

            self._body       = self._world.CreateStaticBody(position=config['pos'])
            self._body_shape = self._body.CreatePolygonFixture(vertices=config['vertices'])
        
        else:
            raise Exception("Unknown type")


    def get_vertices_phys(self):

        vertices = [(self._body.transform * v) for v in self._body_shape.shape.vertices]


        return vertices

    def get_vertices_local(self):

        vertices = self._body_shape.shape.vertices

        return vertices

    def get_vertices(self, cam_pos = (0,0)):

        # The fixture holds information like density and friction,
        # and also the shape.
        shape = self._body_shape.shape

        # Naively assume that this is a polygon shape. (not good normally!)
        # We take the body's transform and multiply it with each
        # vertex, and then convert from meters to pixels with the scale
        # factor.
        vertices = [(self._body.transform * v) * self._ppm for v in shape.vertices]

        # Box2d uses cannonical axis, we need to convert to match the screen axis convention
        # (-y) 
        vertices = [(v[0] - cam_pos[0], self._config['image_height'] - v[1] - cam_pos[1]) for v in vertices]


        return vertices

    def draw(self, surface, cam_pos = (0,0)):

        vertices = self.get_vertices(cam_pos)

        #print vertices

        pygame.draw.polygon(surface, self._colour, vertices)

        cs = [(125,0,0), (0,125,0), (0,0,125), (125,125,125)]
        if self._config['draw_corners']:
            for idx in range(len(vertices)):
                v = vertices[idx]
                c = cs[idx]
                pygame.draw.circle(surface, c, (int(v[0]),int(v[1])), 5, 0)


    def get_angle(self):

        return self._body.angle



    def get_state(self):

        (px,py) = self._body.position
        angle = self._body.angle
        (vx,vy) = self._body.linearVelocity
        omega = self._body.angularVelocity

        state = { 'position': np.array([px,py]), 
                  'angle': angle, 
                  'linear_velocity': np.array([vx,vy]),
                   'angular_velocity': omega
                }

        return state


    def to_vec_world(self, theta):

        return np.array([np.cos(theta + self._body.angle), np.sin(theta + self._body.angle)])


    def get_point(self,local_point):
        p = self._body.transform*local_point

        return np.array([p[0],p[1]])

    def get_image_point(self,world_point):
        px = world_point[0]*self._ppm
        py = self._config['image_height'] - world_point[1]*self._ppm 


        return np.array([px,py])

    def get_image_point2(self,local_point):
        px, py = self._body.transform*local_point*self._ppm
        py = self._config['image_height'] - py

        return np.array([px,py])

    def get_image_point_norm(self,local_point):
        (px,py) = self.get_image_point2(local_point)

        px /= float(self._config['image_width'])
        py /= float(self._config['image_height'])

        return np.array([px,py])

    def get_push_action(self):

        n_vertices = len(self.get_vertices_local())
        step = 1.0/n_vertices
        acc = 0.0
        push_actions = []
        for i in range(n_vertices):
            push_actions.append(np.random.uniform(acc, acc+step))
            acc += step

        return push_actions[np.random.randint(0,len(push_actions))] #np.random.randint(0,4)


    def get_centre_local(self):
        return np.mean(self.get_vertices_local())

    def sample_push_action(self, alpha = None, push_mag = 0.05, pre_push_offset = 0.05):

        if alpha is None:
            alpha = self.get_push_action()

        if isinstance(alpha, np.ndarray):
            alpha = float(alpha[0])
        action = alpha

        vertices = self.get_vertices_local()


        step = 1.0/len(vertices)
        accs = [0.0]
        # find idx
        idx = 0
        # alpha = [0.15,0.35,0.60,0.90][2]
        while idx < len(vertices) and alpha > accs[-1]:
            prev_acc = accs[-1]
            accs.append(prev_acc+step)
            idx += 1

        print accs, idx

        if idx >= len(vertices):
            p0 = np.array(vertices[len(vertices) - 1])  
            p1 = np.array(vertices[0])
        else:
            p0 = np.array(vertices[idx-1])  
            p1 = np.array(vertices[idx])

        alpha = (alpha-accs[idx-1])/step
        

        # if  0. <= alpha <= 0.33:
        #     alpha = alpha/0.33
        #     p0 = np.array(vertices[0])  
        #     p1 = np.array(vertices[1])
        # elif 0.33 < alpha <= 0.66:
        #     alpha = (alpha-0.33)/0.33
        #     p0 = np.array(vertices[1])  
        #     p1 = np.array(vertices[2])
        # elif 0.66 < alpha <= 1.0:
        #     alpha = (alpha-0.66)/0.33
        #     p0 = np.array(vertices[2])  
        #     p1 = np.array(vertices[0])
        # else:
        #     print "Received alpha is \t", alpha
        #     print "This is unacceptable alpha, acceptable range is (0,1) please check the code."

        # self._body_shape._dyn_body.angle = np.pi/6

        edge_point = p0*(1 - alpha) + alpha*p1#np.multiply(np.random.rand(2),[2*box_w,2*box_h]) - np.array([box_w,box_h])

        centre = self.get_centre_local()

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

        f_mag = push_mag# np.random.uniform(0, self._config['push_mag'])

        px, py = edge_point

        pre_push_pos_box_local = edge_point - push_vec*pre_push_offset

        
        pre_push_pos_world = self.get_point(pre_push_pos_box_local)
        push_pos_world     = self.get_point(edge_point)

        fin_push_dir = (push_pos_world - pre_push_pos_world)/np.linalg.norm(pre_push_pos_world - push_pos_world)

        fx = f_mag*fin_push_dir[0]
        fy = f_mag*fin_push_dir[1]

        # print self._push_counter
        return action, pre_push_pos_world[0], pre_push_pos_world[1], fx, fy, push_pos_world[0], push_pos_world[1]



    def apply_push(self, px, py, force_mag=10., theta=0.):

        px_world, py_world = self.get_point((px,py))
        ix, iy = self.to_vec_world(theta)

        self._body.ApplyLinearImpulse(impulse=(ix*force_mag,iy*force_mag), 
                                point=(px_world,py_world), wake=True)

    def apply_push2(self, px, py, fx, fy):

        px_world, py_world = self.get_point((px,py))

        print "px_world py_world", px_world, py_world

        self._body.ApplyLinearImpulse(impulse=(fx,fy), 
                                point=(px_world,py_world), wake=True)


    def apply_push3(self, px, py, fx, fy):

        px_world, py_world = self.get_point((px,py))

        self._body.ApplyForce(force=(fx,fy), 
                                point=(px_world,py_world), wake=True)   

    def reset(self, noise = 0.5):
        body = self._body

        body.position = self._config['pos'] + np.random.randn(2)*np.sqrt(noise)
        body.angle = self._config['ori']
        body.linearVelocity = (0,0)
        body.angularVelocity = 0

    def set_state(self, state):
        '''
        assumes following state order
        state = [x, y, th, dx, dy, dth]
        '''
        body = self._body
        body.position = (state[0], state[1])
        body.angle = state[2]
        body.linearVelocity = (state[3],state[4])
        body.angularVelocity = state[5]