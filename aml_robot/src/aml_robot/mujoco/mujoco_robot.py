
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

        
    def _configure(self, viewer, on_state_callback=None):
        
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        self._kinematics = None

        self._pub_rate = None

        self.set_sampling_rate()

        # self.set_command_timeout(0.2)

        self._camera = viewer

        _update_period = rospy.Duration(1.0/self._sampling_rate)

        rospy.Timer(_update_period, self._update_state)


    def set_sampling_rate(self, rate=100):
        self._sampling_rate = rate


    def _update_state(self, event):

        now                 = rospy.Time.now()

        state = {}
        state['position']        = self._model.data.qpos[0:self._nv].flat
        state['velocity']        = self._model.data.qvel[0:self._nv].flat
        state['effort']          = self._model.data.qfrc_inverse[0:self._nv]
        state['jacobian']        = self._model.jacSite('ee_site')
        state['inertia']         = self._model.fullM()
        state['rgb_image']       = self._camera.get_image()[0]
        # state['depth_image']     = self._camera.curr_depth_image
        state['gravity_comp']    = self._model.data.qfrc_bias[0:self._nv] + self._model.data.qfrc_passive[0:self._nv]

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

        return ee_pose[0], np.quaternion(ee_pose[1][0],ee_pose[1][1],ee_pose[1][2],ee_pose[1][3])

    def get_compensation_forces(self):

        return self._model.data.qfrc_bias[0:self._nu] + self._model.data.qfrc_passive[0:self._nu]

    def get_ee_velocity(self):

        vel = self._model.data.qvel[6:].flat().copy()

        ee_vel = vel[:3]

        ee_omg = vel[3:]

        return ee_vel, ee_omg


    def reset_model(self):
        self._model.resetData()


    #mimicking the untuck arm of baxter_robot
    def untuck_arm(self):

        qpos = self._model.data.qpos.copy()

        qpos[7:] = self._reset_qpos[7:]

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