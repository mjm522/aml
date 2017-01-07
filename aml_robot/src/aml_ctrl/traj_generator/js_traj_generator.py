import numpy as np
import quaternion
from aml_ctrl.traj_generator.traj_generator import TrajGenerator

from aml_lfd.utilities.utilities import get_js_traj
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp

class JSTrajGenerator(TrajGenerator):
    
    def __init__(self, load_from_demo=False, **kwargs):

        TrajGenerator.__init__(self, load_from_demo=load_from_demo, **kwargs)

    def get_demo_traj(self):

        js_pos, js_vel, js_acc = get_js_traj(demo_idx=self._demo_idx)

        self._timesteps = len(js_pos)

        #position trajectory
        self._traj['pos_traj'] = js_pos
        #velocity trajectory
        self._traj['vel_traj'] = js_vel
        #acceleration trajectory
        self._traj['acc_traj'] = js_acc

    def get_traj_interp(self):

        min_jerk_interp = MinJerkInterp()
        
        min_jerk_interp.configure(start_pos=self._start_pos, 
                                  goal_pos=self._goal_pos, 
                                  start_qt=self._start_ori, 
                                  goal_qt=self._goal_ori)

        self._traj = min_jerk_interp.get_interpolated_trajectory()


    def generate_traj(self):

        if self._load_from_demo:

            self.get_demo_traj()

        else:

            self.get_traj_interp()

        return self._traj
