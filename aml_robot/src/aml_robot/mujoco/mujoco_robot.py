
import rospy
import mujoco_py
from mujoco_py import mjcore
from mujoco_py import mjtypes
import numpy as np
import quaternion
import copy

from aml_lfd.utilities.utilities import compute_omg
from aml_robot.utilities.utilities import convert_rospy_time2sec

class MujocoRobot():
    def __init__(self, xml_path): 
            
        #get the path of the xml file
        self._xml_path = xml_path
        self._model = mjcore.MjModel(self._xml_path)
        self._dt = self._model.opt.timestep;
        self._metadata = {'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self._dt))}

        self._nu = self._model.nu

        self._nv = self._model.nv

        self._reset_qpos = copy.deepcopy(self._model.data.qpos)

        
    def _configure(self, viewer, p_start_idx=None, p_end_idx=None, v_start_idx=None, v_end_idx=None, on_state_callback=None):
        
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        self._kinematics = None

        self._pub_rate = None

        self.set_sampling_rate()

        self.robot_state_set_start_end_index(p_start_idx, p_end_idx, v_start_idx, v_end_idx)

        # self.set_command_timeout(0.2)

        self._viewer = viewer

        _update_period = rospy.Duration(1.0/self._sampling_rate)

        rospy.Timer(_update_period, self._update_state)


    def set_sampling_rate(self, rate=100):
        
        self._sampling_rate = rate

    def robot_state_set_start_end_index(self, p_start_idx, p_end_idx, v_start_idx, v_end_idx):

        '''
        since mujoco provides a single array of all states of all objects in the scene, it becomes essential
        to give a start index and end index in that array which corresponds to the robot
        if the input arguments are none, then all values are returned
        '''

        if p_start_idx is None:
            p_start_idx = 0
        if p_end_idx is None:
            p_end_idx = self._nv
        if v_start_idx is None:
            v_start_idx = 0
        if v_end_idx is None:
            v_end_idx = self._nv
        
        self._p_start_idx = p_start_idx
        self._p_end_idx   = p_end_idx
        self._v_start_idx = v_start_idx
        self._v_end_idx = v_end_idx


    def _update_state(self, event):

        now                      = rospy.Time.now()

        state = {}
        state['position']        = self._model.data.qpos[self._p_start_idx:self._p_end_idx].flatten()
        state['velocity']        = self._model.data.qvel[self._p_start_idx:self._p_end_idx].flatten()
        state['effort']          = self._model.data.qfrc_inverse[self._p_start_idx:self._p_end_idx]
        state['jacobian']        = self._model.jacSite('ee_site')
        state['inertia']         = self._model.fullM()
        state['rgb_image']       = self._viewer.get_image()[0]
        # state['depth_image']     = self._camera.curr_depth_image
        state['gravity_comp']    = self._model.data.qfrc_bias[self._p_start_idx:self._p_end_idx] + self._model.data.qfrc_passive[self._p_start_idx:self._p_end_idx]

        state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        try:
            state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        except:
            pass

        try:
            state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
        except:
            pass

        self._state = state


    def get_ee_pose(self):
        
        ee_pose = self._model.site_pose('ee_site')

        return ee_pose[0], np.quaternion(ee_pose[1][3],ee_pose[1][0],ee_pose[1][1],ee_pose[1][2])

    def get_compensation_forces(self):

        return self._model.data.qfrc_bias[0:self._nu] + self._model.data.qfrc_passive[0:self._nu]

    def get_ee_velocity(self):

        vel = self._model.data.qvel[self._v_start_idx:self._v_end_idx].flatten().copy()

        ee_vel = vel[:3]

        ee_omg = vel[3:]

        return ee_vel, ee_omg


    def reset_model(self):
        self._model.resetData()


    #mimicking the untuck arm of baxter_robot
    def untuck_arm(self):

        qpos = self._model.data.qpos.copy()

        qpos[self._p_start_idx:] = self._reset_qpos[self._p_start_idx:]

        self.set_qpos(qpos)


    def set_qpos(self, qpos):
        self._model.data.qpos = qpos

    
    def set_qvel(self, qvel):
        self._model.data.qvel = qvel


    def set_qacc(self, qacc):
        self._model.data.qacc = qacc


    def inv_dyn(self, qpos, qvel, qacc):
        self._model.data.qpos = qpos
        self._model.data.qvel = qvel
        self._model.data.qacc = qacc
        self._model.inverse()
        return self._model.data.qfrc_inverse
        
    def exec_torque_cmd(self, cmd):
        self._model.data.ctrl = cmd
        self._model.step()

    def exec_position_cmd(self, cmd):
        self._model.data.ctrl = cmd
        self._model.step()

    def exec_position_cmd_delta(self, cmd):
        self._model.data.ctrl = cmd + self._model.data.qpos[0:self._nu]
        self._model.step()