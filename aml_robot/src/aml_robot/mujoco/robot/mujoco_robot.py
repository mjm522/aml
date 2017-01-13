import mujoco_py
from mujoco_py import mjcore
from mujoco_py import mjtypes
import numpy as np
import quaternion

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

    def _configure(self, model, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        self._kinematics = None

        self._ik_baxter = None

        self._pub_rate = None

        # self.set_sampling_rate()

        # self.set_command_timeout(0.2)

        self._camera = camera_sensor.CameraSensor()
    
    def _update_state(self):

        now                 = rospy.Time.now()

        state = {}
        state['position']        = self._model.data.qpos[0:self._nu].flat
        state['velocity']        = self._model.data.qvel[0:self._nu].flat
        state['effort']          = self._model.data.qfrc_inverse[0:self._nu]
        state['jacobian']        = self._model.jacSite('ee_site')
        state['inertia']         = self._model.fullM()
        state['rgb_image']       = self._camera.curr_rgb_image
        state['depth_image']     = self._camera.curr_depth_image
        state['gravity_comp']    = self._model.data.qfrc_bias[0:self._nu] + self._model.data.qfrc_passive[0:self._nu]

        state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        try:
            state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        except:
            pass

        try:
            state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
        except:
            pass

        return state

    def get_ee_pose():
        
        ee_pose = self._model.site_pose('ee_site')

        return ee_pose[0], np.quaternion(ee_pose[1][0],ee_pose[1][1],ee_pose[1][2],ee_pose[1][3])

    def get_ee_velocity():

        time_now = rospy.Time.now()

        ee_point, ee_ori  = self.get_ee_pose()

        ee_vel = (ee_point - self._state['ee_point'])/(convert_rospy_time2sec(self._state(timestamp))-\
                                           convert_rospy_time2sec(time_now))

        ee_omg = compute_omg(self._state['ee_ori'], ee_ori)

        return ee_vel, ee_omg


    def move_to_joint_pos(self, cmd):
        self._model.data.qpos = cmd
        self._model._compute_subtree() 
        self._model.forward()
    
    def exec_velocity_cmd(self, cmd):
        self._model.data.qvel = cmd
        self._model._compute_subtree() 
        self._model.forward()
        
    def exec_torque_cmd(self, cmd):
        self._model.data.ctrl = cmd
        self._model.step()

    def exec_position_cmd(self, cmd):
        self._model.data.ctrl = cmd
        self._model.step()

    def exec_position_cmd_delta(self, cmd):
        self._model.data.ctrl = cmd + self._model.data.qpos[0:self._nu]
        self._model.step()