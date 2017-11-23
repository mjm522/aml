import numpy as np
from aml_planners.push_planner.utilities.utils import sigmoid
from aml_planners.push_planner.dynamics.dynamics import Dynamics
from aml_planners.push_planner.push_worlds.box2d_push_world import Box2DPushWorld
from aml_planners.push_planner.exp_params.experiment_params import experiment_config


class Box2DDynamics(Dynamics):

    def __init__(self, config):

        Dynamics.__init__(self, config['push_world_config']['dt'])

        config['push_world_config']['dt'] = self._dt

        self._world  = Box2DPushWorld(config['push_world_config'])
        self._world.set_state([0.,0.,0.,0.,0.,0.])

        self._is_quasi_static = config['is_quasi_static']


    def convert_push_action(self, action):

        # action, theta, f_mag, px, py, fx, fy = self._world.sample_push_action(action)
        action, pre_push_pos_x, pre_push_pos_y, fx, fy, push_pos_x, push_pos_y = self._world.sample_push_action2(action)
        return [[action, pre_push_pos_x, pre_push_pos_y, fx, fy, push_pos_x, push_pos_y]]
        # return [px,py,fx,fy]


    def dynamics(self, x, u):

        self._world.set_state(x)
        pc = 0
        ## Send action to world
        
        # action = [0., 0., u[0], u[1]] #px, py, f_x, f_y

        action = self.convert_push_action(sigmoid(u)) #

        # print "Action is \t", action

        self._world.update(action)

        ## Step the world for certain number of time steps
        # for i in range(self._number_steps):       
        while pc <= 1:#10:
            self._world.step()
            pc += 1
        # pass

        final_state = self._world.pack_box_state()

        if self._is_quasi_static:
            final_state[3:] = 0.

        self._world.reset()

        return final_state, 0.0, action # uncertainty is zero


    def visualize(self, box_pose, fin_pose):
        pass




###########TEST CODE#######################
def main():

    bxDyn = Box2DDynamics(experiment_config)
    state = [16.,0.,0.,0.,0.,0.,0.]
    u = [0.,0.,1.,np.pi]

    for k in range(100):
        print state
        state, _ = bxDyn.dynamics(state,u)

if __name__ == '__main__':
    main()

