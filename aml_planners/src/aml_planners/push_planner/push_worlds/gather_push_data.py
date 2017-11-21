import os
import data_vis
from config import config

import numpy as np
import pandas as pd

np.random.seed(42)

from push_world import PushWorld

class DataCollector(object):


    def __init__(self, world, initial_box_states, pushes_per_location, n_seq_pushes = 2, noise = 0.1):


        self._noise = noise


        self._world = world

        self._x0s = initial_box_states

        self._ppl = pushes_per_location

        self._n_seq_pushes = n_seq_pushes

        self._total_pushes = len(self._x0s)*self._ppl


        self._df = pd.DataFrame(columns=['xi','yi','thetai',
                               'action',
                               'xf','yf','thetaf'])


    def collect_push(self):

        state0 = self._world.pack_box_state()

        ## Controller selects push action (random sampling for simplicity now)
        action = self._world.sample_push_action2()

        ## Send action to world
        self._world.update(action)


        ## Step the world for certain number of time steps
        for i in range(config['steps_per_frame']):       
            self._world.step()


        statef = self._world.pack_box_state()

        rot_matrix = np.array([[0.,  1.],[-1., 0.]])
        state_0_trans = np.dot(rot_matrix, state0[:2])
        theta_0_trans = (state0[2] - np.pi/2)%(2*np.pi)
        state_f_trans = np.dot(rot_matrix, statef[:2])
        theta_f_trans = (statef[2] - np.pi/2)%(2*np.pi)

        # tmp = pd.DataFrame(np.reshape(np.r_[state_0_trans,theta_0_trans, action[0], state_f_trans, theta_f_trans],(1,7)), columns=['xi','yi','thetai',
        #                                                                             'action',
        #                                                                             'xf','yf','thetaf'])

        tmp = pd.DataFrame(np.reshape(np.r_[state0[:3],action[0],statef[:3]],(1,7)), columns=['xi','yi','thetai',
                                                                                    'action',
                                                                                    'xf','yf','thetaf'])
        
        self._df = self._df.append(tmp, ignore_index=True)

        # print df.size



        # if pc >= 5:
        #     action = world.sample_push_action()
        #     # world.reset()




    def collect_data(self):


        push_counter = 0
        
        while push_counter < self._total_pushes:

            for x0 in self._x0s:

                for i in range(self._ppl):
                    
                    self._world.reset()
                    rnd = np.random.randn(6)*self._noise
                    tmp_x0 = x0.copy()
                    # tmp_x0 += rnd
                    
                    tmp_x0[:2] += rnd[:2]
                    # tmp_x0[3:5] += rnd[3:5]

                    self._world.set_state(tmp_x0)

                    for k in range(self._n_seq_pushes):
                        self.collect_push()

                        push_counter += 1

                    # print "Push Counter: ", self._df.shape[0], push_counter




    def save_csv(self, filename=os.environ['MPPI_DATA_DIR'] +'push_data.csv'):

        self._df.to_csv(filename)


def main():

    # dt = 
    world = PushWorld(config)

    filename = os.environ['MPPI_DATA_DIR'] + 'box2d_push_data_all_rand_circle.csv'

    ys = np.linspace(0., 7., 10)
    xs = np.linspace(0., 7., 10)


    # #data with missing 6 blocks
    # x0s = [np.array([0.959, 0.739, 0.,0.,0.,0.]),
    #        np.array([0.822, 0.737, 0.,0.,0.,0.]),
    #        np.array([0.701, 0.734, 0.,0.,0.,0.]),
    #        np.array([0.678, 0.732, 0.,0.,0.,0.]),
    #        np.array([0.575, 0.607, 0.,0.,0.,0.]),
    #        np.array([0.586, 0.474, 0.,0.,0.,0.]),
    #        np.array([0.587, 0.344, 0.,0.,0.,0.]),
    #        np.array([0.599, 0.223, 0.,0.,0.,0.]),
    #        np.array([0.718, 0.231, 0.,0.,0.,0.]),
    #        np.array([0.841, 0.243, 0.,0.,0.,0.]),
    #        np.array([0.962, 0.110, 0.,0.,0.,0.])]


    # #missing 6 blocks
    # gap_x0s = [np.array([0.839, 0.612, 0.,0.,0.,0.]),
    #            np.array([0.835, 0.491, 0.,0.,0.,0.]),
    #            np.array([0.841, 0.364, 0.,0.,0.,0.]),
    #            np.array([0.705, 0.611, 0.,0.,0.,0.]),
    #            np.array([0.713, 0.483, 0.,0.,0.,0.]),
    #            np.array([0.714, 0.355, 0.,0.,0.,0.])]

    # x0s += gap_x0s

    x0s = []
    for x in xs:
        for y in ys:
            
            if (x-3.5)**2 + (y-3.5)**2 - 4 < 0.: 
                continue

            x0 = np.array([x,y,0.,0.,0.,0.])
            x0s.append(x0.copy())

    #[5, 20, 0, 0, 0, 0]


    dc = DataCollector(world, initial_box_states = x0s, pushes_per_location = 20, n_seq_pushes = 1, noise = 0.2)#0.0001

    dc.collect_data()

    dc.save_csv(filename)

    data_vis.visualise_push_data(filename)


if __name__ == '__main__':
    main()