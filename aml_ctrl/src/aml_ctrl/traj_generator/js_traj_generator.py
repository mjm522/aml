import numpy as np
import quaternion
from aml_ctrl.traj_generator.traj_generator import TrajGenerator

from aml_lfd.utilities.utilities import get_js_traj

from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp

class JSTrajGenerator(TrajGenerator):
    
    def __init__(self, load_from_demo=False, **kwargs):

        TrajGenerator.__init__(self, load_from_demo=load_from_demo, **kwargs)

        self._robot = None

    def configure(self, robot_interface):

        self._robot =  robot_interface

    def get_demo_traj(self):

        if self._demo_path is not None:

            js_pos, js_vel, js_acc = get_js_traj(limb_name=self._limb_name, demo_path=self._demo_path)

        else:

            js_pos, js_vel, js_acc = get_js_traj(limb_name=self._limb_name, demo_idx=self._demo_idx)

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

        return min_jerk_interp.get_interpolated_trajectory()

    def convert_os_to_js(self, os_traj=None):

        if self._robot is None:

            print "Inverse Kinematics client not configured, please configure ..."
            raise ValueError

        if os_traj is None:

            os_traj = self.get_traj_interp()

        n_steps = len(os_traj['ori_traj'])

        js_traj = []

        for t in range(n_steps):
            
            goal_pos = os_traj['pos_traj'][t]
            goal_ori = quaternion.as_float_array(os_traj['ori_traj'][t])[0]

            if np.any(np.isnan(goal_pos)) or np.any(np.isnan(goal_ori)):
                
                pass

            else:

                success, limb_joints = self._robot.ik(pos=goal_pos, ori=goal_ori)

                # print "limb joints \t", np.round(limb_joints ,3)

                if success:

                    js_traj.append(limb_joints)

                else:

                    print "Failed to find IK solution for pose on index \t", t


        #position trajectory
        self._traj['pos_traj'] = np.asarray(js_traj).squeeze()
        #velocity trajectory
        self._traj['vel_traj'] = None
        #acceleration trajectory
        self._traj['acc_traj'] = None


    def generate_traj(self):

        if self._load_from_demo:

            self.get_demo_traj()

        else:

            self.convert_os_to_js()

        return self._traj
