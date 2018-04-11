import numpy as np
import quaternion
from aml_ctrl.traj_generator.traj_generator import TrajGenerator
from aml_lfd.utilities.utilities import get_os_traj
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp

class OSTrajGenerator(TrajGenerator):
    
    def __init__(self, load_from_demo=False, **kwargs):

        TrajGenerator.__init__(self, load_from_demo=load_from_demo, **kwargs)

    def get_demo_traj(self):

        if self._demo_path is not None:

            demo_idx = None
        else:
            demo_idx = self._demo_idx

        os_pos_traj, os_ori_traj, os_vel_traj, \
        os_omg_traj, os_acc_traj, os_ang_traj = get_os_traj(limb_name=self._limb_name, 
                                                            demo_idx=demo_idx, 
                                                            file_path=self._demo_path)

        self._timesteps = len(os_pos_traj)

        #position trajectories

        #position trajectory
        self._traj['pos_traj'] = os_pos_traj
        #velocity trajectory
        self._traj['vel_traj'] = os_vel_traj
        #acceleration trajectory
        self._traj['acc_traj'] = os_acc_traj

        #orientation trajectories

        #position trajectory
        self._traj['ori_traj'] = os_ori_traj
        #velocity trajectory
        self._traj['omg_traj'] = os_omg_traj
        #acceleration trajectory
        self._traj['ang_traj'] = os_ang_traj


    def get_traj_interp(self):
        
        min_jerk_interp = MinJerkInterp()
        
        min_jerk_interp.configure(start_pos=self._start_pos, 
                                  goal_pos=self._goal_pos, 
                                  start_qt=self._start_ori, 
                                  goal_qt=self._goal_ori)

        min_jerk_traj = min_jerk_interp.get_interpolated_trajectory()

        self._traj = min_jerk_interp.get_interpolated_trajectory()


    def generate_traj(self):

    	if self._load_from_demo:

    		self.get_demo_traj()

        else:

            self.get_traj_interp()

        return self._traj
