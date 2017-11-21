import data_vis
import numpy as np
import pandas as pd

np.random.seed(42)

class DataCollector(object):

    def __init__(self, world, config, initial_box_states, pushes_per_location, n_seq_pushes = 2, noise = 0.1):

        self._noise = noise

        self._world = world

        self._config = config

        self._x0s = initial_box_states

        self._ppl = pushes_per_location

        self._n_seq_pushes = n_seq_pushes

        self._total_pushes = len(self._x0s)*self._ppl

        self._df = pd.DataFrame(columns=['xi','yi','thetai','dxi','dyi','dthetai',
                               'a_px', 'a_py', 'a_fx', 'a_fy',
                               'xf','yf','thetaf','dxf','dyf','dthetaf'])


    def collect_push(self):

        state0 = self._world.pack_box_state()

        ## Controller selects push action (random sampling for simplicity now)
        action = self._world.sample_push_action()

        ## Send action to world
        self._world.update(action)


        ## Step the world for certain number of time steps
        for i in range(self._config['steps_per_frame']):       
            self._world.step()


        statef = self._world.pack_box_state()

        tmp = pd.DataFrame(np.reshape(np.r_[state0,action,statef],(1,16)), columns=['xi','yi','thetai','dxi','dyi','dthetai',
                                                              'a_px', 'a_py', 'a_fx', 'a_fy',
                                                              'xf','yf','thetaf','dxf','dyf','dthetaf'])
        
        self._df = self._df.append(tmp, ignore_index=True)


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

    def save_csv(self, filename):

        self._df.to_csv(filename)