import numpy as np
import quaternion
from aml_lfd.utilities.utilities import get_js_traj, plot_demo_data

class JSTrajGenerator():
    
    def __init__(self, dt=0.05, tau=5.):
        self._dt = dt
        self._tau = tau
        self._timesteps = np.arange(0, 2*self._tau, self._dt)
        self._js_traj = {}

    def configure(self, demo_idx=1, start_pos=None, start_qt=None, goal_pos=None, goal_qt=None):
        
        self._demo_idx = demo_idx
        # self._start_pos = start_pos
        # self._goal_pos  = goal_pos
        # if isinstance(start_qt, np.quaternion):
        #     start_qt = quaternion.as_float_array(start_qt)[0]
        # if isinstance(goal_qt, np.quaternion):
        #     goal_qt = quaternion.as_float_array(goal_qt)[0]
        # self._start_qt  = start_qt
        # self._goal_qt   = goal_qt

    def osc_to_jsc_traj_conversion(self, osc_traj):
        #TODO, we could find a closed loop IK service based JSC trajectory from eetrajectory
        pass

    def jsc_traj_from_demonstration(self, demo_idx=1):

        return get_js_traj(demo_idx=demo_idx)


    def get_interpolated_trajectory(self):
        
        js_pos, js_vel, js_acc = self.jsc_traj_from_demonstration(demo_idx=self._demo_idx)

        self._timesteps = len(js_pos)

        #position trajectory
        self._js_traj['pos_traj'] = js_pos
        #velocity trajectory
        self._js_traj['vel_traj'] = js_vel
        #acceleration trajectory
        self._js_traj['acc_traj'] = js_acc

        # plot_demo_data(demo_idx=self._demo_idx)

        return self._js_traj