import roslib
roslib.load_manifest('aml_robot')

import rospy

import baxter_interface
import baxter_external_devices

from std_msgs.msg import (
    UInt16,
)
from baxter_core_msgs.msg import SEAJointState
from baxter_interface import CHECK_VERSION
from baxter_kinematics import baxter_kinematics

import numpy as np
import quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from aml_perception import camera_sensor 

#for computation of angular velocity
from aml_lfd.utilities.utilities import compute_omg

from aml_robot.baxter_ik import IKBaxter

#from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

class BaxterArm(baxter_interface.limb.Limb):

    def __init__(self, limb, on_state_callback=None):

        self._ready = False

        self._configure(limb,on_state_callback)
        
        self._ready = True

        #number of joints
        self._nq = 7
        #number of control commads
        self._nu = 7

        self._limb = limb

        #these values are from the baxter urdf file
        self._jnt_limits = [{'lower':-1.70167993878,  'upper':1.70167993878},
                            {'lower':-2.147,          'upper':1.047},
                            {'lower':-3.05417993878,  'upper':3.05417993878},
                            {'lower':-0.05,           'upper':2.618},
                            {'lower':-3.059,          'upper':3.059},
                            {'lower':-1.57079632679,  'upper':2.094},
                            {'lower':-3.059,          'upper':3.059}]

        if limb == 'left':
            #secondary goal for the manipulator
            self._limb_group = 0
            self.q_mean  = np.array([ 0.0,  -0.55,  0.,   1.284, 0.,   0.262, 0.])
            self._tuck   = np.array([-1.0,  -2.07,  3.0,  2.55,  0.0,  0.01,  0.0])
            self._untuck = np.array([-0.08, -1.0,  -1.19, 1.94,  0.67, 1.03, -0.50])
        
        elif limb == 'right':
            self._limb_group = 1
            self.q_mean  = np.array([0.0,  -0.55,  0.,   1.284,  0.,   0.262, 0.])
            self._tuck   = np.array([1.0,  -2.07, -3.0,  2.55,   0.0,  0.01,  0.0])
            self._untuck = np.array([0.08, -1.0,   1.19, 1.94,  -0.67, 1.03,  0.50])
                                                            
        else:
            print "Unknown limb idex"
            raise ValueError

        self._cuff = baxter_interface.DigitalIO('%s_lower_cuff' % (limb,))
        
        self._cuff.state_changed.connect(self.cuff_cb)

        self._cuff_state = None
            
        baxter_interface.RobotEnable(CHECK_VERSION).enable()

        #this will be useful to compute ee_velocity using finite differences
        self._ee_pos_old, self._ee_ori_old = self.get_ee_pose()
        self._time_now_old = self.get_time_in_seconds()

    def cuff_cb(self, value):
        self._cuff_state = value

    #this function returns self._cuff_state to be true
    #when arm is moved by a demonstrator, the moment arm stops 
    #moving, the status returns to false
    #initial value of the cuff is None, it is made False by pressing the
    #cuff button once
    @property
    def get_lfd_status(self):
        return self._cuff_state

    def set_sampling_rate(self, sampling_rate=100):
        self._pub_rate.publish(sampling_rate)

    def tuck_arm(self):
        self.move_to_joint_pos(self._tuck)

    def untuck_arm(self):
        self.move_to_joint_pos(self._untuck)

    def _configure(self, limb, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        # Parent constructor
        baxter_interface.limb.Limb.__init__(self, limb)

        self._kinematics = baxter_kinematics(self)

        self._ik_baxter = IKBaxter(limb=self)

        self._ik_baxter.configure_ik_service()

        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)

        self._gravity_comp = rospy.Subscriber('robot/limb/' + limb + '/gravity_compensation_torques', 
                                               SEAJointState, self._gravity_comp_callback)
        #gravity + feed forward torques
        self._h = [0. for _ in range(7)]

        self.set_sampling_rate()

        self.set_command_timeout(0.2)

        self._camera = camera_sensor.CameraSensor()

    def _gravity_comp_callback(self, msg):
        self._h = msg.gravity_model_effort
        # print "commanded_effort \n",   msg.commanded_effort
        # print "commanded_velocity \n", msg.commanded_velocity
        # print "commanded_position \n", msg.commanded_position
        # print "actual_position \n", msg.actual_position
        # print "actual_velocity \n", msg.actual_velocity
        # print "actual_effort \n", msg.actual_effort
        # print "gravity_model_effort \n", msg.gravity_model_effort
        # print "hysteresis_model_effort \n", msg.hysteresis_model_effort
        # print "crosstalk_model_effort \n", msg.crosstalk_model_effort
        # print "difference effort \n", np.array(msg.commanded_effort) - np.array(msg.actual_effort)
        # print "difference velocity \n", np.array(msg.commanded_velocity) - np.array(msg.actual_velocity)
        # print "difference position \n", np.array(msg.commanded_position) - np.array(msg.actual_position)


    def _update_state(self):

        now                 = rospy.Time.now()

        joint_angles        = self.joint_angles()
        joint_velocities    = self.joint_velocities()
        joint_efforts       = self.joint_efforts()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        state = {}
        state['position']        = np.array(to_list(joint_angles))
        state['velocity']        = np.array(to_list(joint_velocities))
        state['effort']          = np.array(to_list(joint_efforts))
        state['jacobian']        = self.get_jacobian_from_joints(None)
        state['inertia']         = self.get_arm_inertia(None)
        state['rgb_image']       = self._camera._curr_rgb_image
        state['depth_image']     = self._camera._curr_depth_image
        state['gravity_comp']    = np.array(self._h)


        state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        try:
            state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        except:
            pass

        try:
            state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
        except:
            pass

        # ee_velocity              = self.endpoint_velocity()['linear']
        # state['ee_velocity']     = np.array([ee_velocity.x, ee_velocity.y, ee_velocity.z])

        return state

    def angles(self):
        joint_angles        = self.joint_angles()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        return np.array(to_list(joint_angles))
        

    def _on_joint_states(self, msg):
        baxter_interface.limb.Limb._on_joint_states(self,msg)

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)


    def get_end_effector_link_name(self):
        return self._kinematics._tip_link

    
    def get_base_link_name(self):
        return self._kinematics._base_link

    
    def exec_position_cmd(self, cmd):
        #there is some issue with this function ... move_to_joint_pos works far better.
        
        curr_q = self._state['position']

        joint_command = dict(zip(self.joint_names(), cmd))

        self.set_joint_positions(joint_command)

    def exec_position_cmd_delta(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()
        joint_command = dict([(joint, curr_q[joint] + cmd[i]) for i, joint in enumerate(joint_names)])
        self.set_joint_positions(joint_command)

    def move_to_joint_pos_delta(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()

        joint_command = dict([(joint, curr_q[joint] + cmd[i]) for i, joint in enumerate(joint_names)])

        self.move_to_joint_positions(joint_command)

    def move_to_joint_pos(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()

        joint_command = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])

        self.move_to_joint_positions(joint_command)

    def exec_velocity_cmd(self,cmd):
        
        joint_names = self.joint_names()

        #can't we use dict(zip(joint_names, cmd)) to combine two lists?
        velocity_command = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])
        
        self.set_joint_velocities(velocity_command)

    def exec_torque_cmd(self,cmd):

        joint_names     = self.joint_names()

        torque_command  = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])
        
        self.set_joint_torques(torque_command)

    def move_to_joint_position(self, joint_angles):
        self.move_to_joint_positions(dict(zip(self.joint_names(),joint_angles)))
    
    def get_ee_pose(self):
        
        ee_point        = self.endpoint_pose()['position']
        ee_point        = np.array([ee_point.x, ee_point.y, ee_point.z])
        
        ee_ori          = self.endpoint_pose()['orientation']
        ee_ori          = np.quaternion(ee_ori.w, ee_ori.x, ee_ori.y, ee_ori.z)
        
        return ee_point, ee_ori

    def get_time_in_seconds(self):
        time_now =  rospy.Time.now()
        return time_now.secs + time_now.nsecs*1e-9

    def get_ee_velocity(self, real_robot=True):
        #this is a simple finite difference based velocity computation
        #please note that this might produce a bug since self._goal_ori_old gets 
        #updated only if get_ee_vel is called. 
        #TODO : to update in get_ee_pose or find a better way to compute velocity
        
        if real_robot:
            
            ee_velocity = self.endpoint_velocity()['linear']
            ee_vel      = np.array([ee_velocity.x, ee_velocity.y, ee_velocity.z])
            ee_omega    = self.endpoint_velocity()['angular']
            ee_omg      = np.array([ee_omega.x, ee_omega.y, ee_omega.z])

        else:

            time_now_new = self.get_time_in_seconds()
            
            ee_pos_new, ee_ori_new = self.get_ee_pose()  

            dt = time_now_new-self._time_now_old

            ee_vel = (ee_pos_new - self._ee_pos_old)/dt

            ee_omg = compute_omg(ee_ori_new, self._ee_ori_old)/dt

            self._goal_ori_old = ee_ori_new
            self._goal_pos_old = ee_pos_new
            self._time_now_old = time_now_new
        
        return ee_vel, ee_omg


    def get_cartesian_pos_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        pose = np.array(self._kinematics.forward_position_kinematics(argument))
        position = pose[0:3][:,None] #senting as  column vector
        
        w = pose[6]; x = pose[3]; y = pose[4]; z = pose[5] #quarternions
        
        #formula for converting quarternion to rotation matrix

        rotation = np.array([[1.-2.*(y**2+z**2),    2.*(x*y-z*w),           2.*(x*z+y*w)],\
                             [2.*(x*y+z*w),         1.-2.*(x**2+z**2),      2.*(y*z-x*w)],\
                             [2.*(x*z-y*w),         2.*(y*z+x*w),           1.-2.*(x**2+y**2)]])
        
        return position, rotation

    def get_cartesian_vel_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        return np.array(self._kinematics.forward_velocity_kinematics(argument))[0:3] #only position

    def get_jacobian_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        return np.array(self._kinematics.jacobian(argument))


    def get_arm_inertia(self, joint_angles=None):
        if joint_angles is None:
            argument = None
        else:
            argument = dict(zip(self.joint_names(),joint_angles))

        return np.array(self._kinematics.inertia(argument))

    def ik(self, pos, ori=None):

        success, soln =  self._ik_baxter.ik_servive_request(pos=pos, ori=ori)

        return success, soln


class BaxterButtonStatus():
    
    def __init__(self):
        self.left_cuff_btn   = baxter_interface.DigitalIO('left_lower_cuff')
        self.left_dash_btn   = baxter_interface.DigitalIO('left_upper_button')
        self.left_circle_btn = baxter_interface.DigitalIO('left_lower_button')

        self.right_cuff_btn   = baxter_interface.DigitalIO('right_lower_cuff')
        self.right_dash_btn   = baxter_interface.DigitalIO('right_upper_button')
        self.right_circle_btn = baxter_interface.DigitalIO('right_lower_button')

        # DigitalIO('<limb_name>_upper_button' )  # 'dash' btn
        # DigitalIO('<limb_name>_lower_button')   # 'circle' btn
        # DigitalIO('<limb_name>_lower_cuff')    # cuff squeeze

        #it pass the cuff sensor state to _cuff_cb
        self.left_cuff_btn.state_changed.connect(self.left_cuff_callback) 
                                                        
        #it pass the dash button state to _dash_cb
        self.left_dash_btn.state_changed.connect(self.left_dash_callback) 
                                                        
        self.left_circle_btn.state_changed.connect(self.left_circle_callback)

        #it pass the cuff sensor state to _cuff_cb
        self.right_cuff_btn.state_changed.connect(self.right_cuff_callback) 
                                                        
        #it pass the dash button state to _dash_cb
        self.right_dash_btn.state_changed.connect(self.right_dash_callback) 
                                  
        #it pass the circle button state to _circle_cb                      
        self.right_circle_btn.state_changed.connect(self.right_circle_callback)
        
        self.left_gripper  = baxter_interface.Gripper('left', CHECK_VERSION)
        self.right_gripper = baxter_interface.Gripper('right', CHECK_VERSION)                                              

        self.left_cuff_btn_state = False
        self.left_dash_btn_state = False
        self.left_dash_btn_value = False
        self.left_circle_btn_state = False

        self.right_cuff_btn_state = False
        self.right_dash_btn_state = False
        self.right_dash_btn_value = False
        self.right_circle_btn_state = False

    def left_cuff_callback(self, value):
        self.left_cuff_btn_state = value

    def left_dash_callback(self, value):
        self.left_dash_btn_value = value
        if self.left_dash_btn_value:
            if self.left_dash_btn_state:
                self.left_dash_btn_state = False
            else:
                self.left_dash_btn_state = True

    def left_circle_callback(self, value):
        if value and not self.left_dash_btn_value:
            if self.left_circle_btn_state:
                self.left_gripper.open()
                self.left_circle_btn_state = False
            else:
                self.left_gripper.close()
        self.left_circle_btn_state = True

    def right_cuff_callback(self, value):
        self.right_cuff_btn_state = value

    def right_dash_callback(self, value):
        self.right_dash_btn_value = value
        if self.right_dash_btn_value:
            if self.right_dash_btn_state:
                self.right_dash_btn_state = False
            else:
                self.right_dash_btn_state = True

    def right_circle_callback(self, value):
        if value and not self.right_dash_btn_value:
            if self.right_circle_btn_state:
                self.right_gripper.open()
                self.right_circle_btn_state = False
            else:
                self.right_gripper.close()
        self.right_circle_btn_state = True

