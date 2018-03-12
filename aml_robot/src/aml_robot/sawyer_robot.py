#!/usr/bin/env python

import roslib
roslib.load_manifest('aml_robot')

import rospy

import numpy as np
import quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import (
    UInt16,
)

import intera_interface
from intera_interface import CHECK_VERSION
import intera_external_devices

from intera_core_msgs.msg import SEAJointState

from aml_robot.sawyer_kinematics import sawyer_kinematics
from aml_robot.sawyer_ik import IKSawyer

from aml_perception import camera_sensor 

#for computation of angular velocity
from aml_math.quaternion_utils import compute_omg

from aml_visual_tools.load_aml_logo import load_aml_logo



#from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

class SawyerArm(intera_interface.Limb):

    def __init__(self, limb = "right", on_state_callback=None):

        #Load aml_logo
        load_aml_logo("/robot/head_display")

        #these values are from the baxter urdf file
        self._jnt_limits = [{'lower':-1.70167993878,  'upper':1.70167993878},
                            {'lower':-2.147,          'upper':1.047},
                            {'lower':-3.05417993878,  'upper':3.05417993878},
                            {'lower':-0.05,           'upper':2.618},
                            {'lower':-3.059,          'upper':3.059},
                            {'lower':-1.57079632679,  'upper':2.094},
                            {'lower':-3.059,          'upper':3.059}]

        #number of joints
        self._nq = len(self._jnt_limits)
        #number of control commads
        self._nu = len(self._jnt_limits)

        self._ready = False

        self._limb = limb

        self._configure(limb,on_state_callback)
        
        self._ready = True

        self._q_mean = np.array([ 0.5*(limit['lower'] + limit['upper']) for limit in self._jnt_limits])

        if self._gripper is not None:
            self._jnt_limits.append({'lower': self._gripper.MIN_POSITION, 'upper': self._gripper.MAX_POSITION})


        self._tuck   = np.array([-3.31223050e-04,-1.18001699e+00,-8.22146399e-05, 2.17995802e+00, -2.70787321e-03,5.69996851e-01, 3.32346747e+00, 2.07798000e-02])
        self._untuck = self._tuck

        if limb == 'left':
            #secondary goal for the manipulator
            self._limb_group = 0
        elif limb == 'right':
            self._limb_group = 1                                          
        else:
            print "Unknown limb idex"
            raise ValueError


            
        intera_interface.RobotEnable(CHECK_VERSION).enable()

        #this will be useful to compute ee_velocity using finite differences
        self._ee_pos_old, self._ee_ori_old = self.get_ee_pose()
        self._time_now_old = self.get_time_in_seconds()

    def _configure_cuff(self):

        self._has_cuff = True

        try:

            self._cuff_state = None


            self._cuff = intera_interface.Cuff(limb=self._limb)
            # connect callback fns to signals
            self._lights = None
            self._lights = intera_interface.Lights()
            self._cuff.register_callback(self._light_action,'{0}_cuff'.format(self._limb))




        except Exception as e:
            print e
            self._has_cuff = False


    def _configure_gripper(self):

        try: 
            self._gripper = intera_interface.Gripper(self._limb)
            if not (self._gripper.is_calibrated() or self._gripper.calibrate() == True):
                rospy.logerr("({0}_gripper) calibration failed.".format(self._gripper.name))
                raise

            if self._has_cuff:
                self._cuff.register_callback(self._close_action,'{0}_button_upper'.format(self._limb))
                self._cuff.register_callback(self._open_action,'{0}_button_lower'.format(self._limb))
            
            rospy.loginfo("{0} Cuff Control initialized...".format(self._gripper.name))

        except Exception as e:
            self._gripper = None


    def _open_action(self, value):
        if value and self._gripper.is_ready():
            rospy.logdebug("gripper open triggered")
            self._gripper.open()
            if self._lights:
                self._set_lights('red', False)
                self._set_lights('green', True)

    def _close_action(self, value):
        if value and self._gripper.is_ready():
            rospy.logdebug("gripper close triggered")
            self._gripper.close()
            if self._lights:
                self._set_lights('green', False)
                self._set_lights('red', True)

    def _light_action(self, value):
        if value:
            rospy.logdebug("cuff grasp triggered")
        else:
            rospy.logdebug("cuff release triggered")
        if self._lights:
            self._set_lights('red', False)
            self._set_lights('green', False)
            self._set_lights('blue', value)

    def _set_lights(self, color, value):
        self._lights.set_light_state('head_{0}_light'.format(color), on=bool(value))
        self._lights.set_light_state('{0}_hand_{1}_light'.format(self._limb, color),on=bool(value))

    def cuff_cb(self, value):
        if self._has_cuff:
            self._cuff_state = value
        else:
            print "CUFF NOT DETECTED"

    #this function returns self._cuff_state to be true
    #when arm is moved by a demonstrator, the moment arm stops 
    #moving, the status returns to false
    #initial value of the cuff is None, it is made False by pressing the
    #cuff button once
    @property
    def get_lfd_status(self):
        if self._has_cuff:
            return self._cuff.cuff_button()
        else:
            print "CUFF NOT DETECTED"
            return None

    def set_sampling_rate(self, sampling_rate=100):
        self._pub_rate.publish(sampling_rate)

    def tuck_arm(self):
        print "NOT IMPLEMENTED"

    def untuck_arm(self):
        self.move_to_neutral()

        print self.angles()

    def _configure(self, limb, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        # Parent constructor
        intera_interface.Limb.__init__(self, limb)

        self._kinematics = sawyer_kinematics(self)

        self._ik_sawyer = IKSawyer(limb=self)

        self._ik_sawyer.configure_ik_service()

        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)

        self._gravity_comp = rospy.Subscriber('robot/limb/' + limb + '/gravity_compensation_torques', 
                                               SEAJointState, self._gravity_comp_callback)
        #gravity + feed forward torques
        self._h = [0. for _ in range(self._nq)]

        self.set_sampling_rate()

        self.set_command_timeout(0.2)

        self._camera = camera_sensor.CameraSensor()

        self._gripper = None
        self._cuff = None

        self._configure_cuff()
        self._configure_gripper()

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

        joint_angles        = self.angles()
        joint_velocities    = self.velocities()
        joint_efforts       = self.joint_efforts()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        state = {}
        state['position']        = joint_angles
        state['velocity']        = joint_velocities
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

        state['gripper_state'] = self.get_gripper_state()

        # ee_velocity              = self.endpoint_velocity()['linear']
        # state['ee_velocity']     = np.array([ee_velocity.x, ee_velocity.y, ee_velocity.z])

        return state

    def angles(self, include_gripper = False):
        joint_angles        = self.joint_angles()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        all_angles = to_list(joint_angles)

        if include_gripper and self._gripper is not None:
            all_angles.append(self._gripper.get_position())
        return np.array(all_angles)

    def velocities(self, include_gripper = False):
        joint_velocities        = self.joint_velocities()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        all_velocities = to_list(joint_velocities)

        if include_gripper and self._gripper is not None:
            all_velocities.append(0.0)
        return np.array(all_velocities)


    def q_mean(self):
        return self._q_mean


    def n_cmd(self):
        return self._nu

    def n_joints(self):
        return self._nq

    def get_state(self):
        return self._state


    def get_gripper_state(self):
        gripper_state = {}

        if self._gripper is not None:
            gripper_state['position'] = self._gripper.get_position()
            gripper_state['force'] = self._gripper.get_force()
            

        return gripper_state


    def set_gripper_speed(self, speed):

         if self._gripper is not None:

            self._gripper.set_velocity(speed)

    def set_arm_speed(self,speed):

        self.set_joint_position_speed(speed)

        

    def _on_joint_states(self, msg):
        intera_interface.Limb._on_joint_states(self,msg)

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)


    def get_end_effector_link_name(self):
        return self._kinematics._tip_link

    
    def get_base_link_name(self):
        return self._kinematics._base_link

    def exec_gripper_cmd(self, pos, force = None):

        if self._gripper is None:
            return

        if force is not None:
            holding_force = min(max(self._gripper.MIN_FORCE,force),self._gripper.MAX_FORCE)

            self._gripper.set_holding_force(holding_force)

        position = min(self._gripper.MAX_POSITION,max(self._gripper.MIN_POSITION,pos))

        self._gripper.set_position(pos)


    def exec_gripper_cmd_delta(self, pos_delta, force_delta = None):

        if self._gripper is None:
            return

        if force_delta is not None:
            force = self._gripper.get_force()
            holding_force = min(max(self._gripper.MIN_FORCE,force+force_delta),self._gripper.MAX_FORCE)

            self._gripper.set_holding_force(holding_force)

        pos = self._gripper.get_position()
        position = min(self._gripper.MAX_POSITION,max(self._gripper.MIN_POSITION,pos + pos_delta))
        
        self._gripper.set_position(position)

    def exec_position_cmd(self, cmd):
        #there is some issue with this function ... move_to_joint_pos works far better.
        
        curr_q = self._state['position']

        if len(cmd) > 7:
            gripper_cmd = cmd[7:]
            self.exec_gripper_cmd(*gripper_cmd)

        joint_command = dict(zip(self.joint_names(), cmd[:7]))

        self.set_joint_positions(joint_command)

    def exec_position_cmd_delta(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()

        joint_command = dict([(joint, curr_q[joint] + cmd[i]) for i, joint in enumerate(joint_names)])
        self.set_joint_positions(joint_command)

        if len(cmd) > 7:
            gripper_cmd = cmd[7:]
            self.exec_gripper_cmd_delta(*gripper_cmd)

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
        jacobian = np.array(self._kinematics.jacobian(argument))

        return jacobian


    def get_arm_inertia(self, joint_angles=None):
        if joint_angles is None:
            argument = None
        else:
            argument = dict(zip(self.joint_names(),joint_angles))

        return np.array(self._kinematics.inertia(argument))

    def ik(self, pos, ori=None):

        success, soln =  self._ik_sawyer.ik_servive_request(pos=pos, ori=ori)

        return success, soln



def main():
    rospy.init_node("sawyer_arm_example")

    arm = SawyerArm('right')

    rospy.spin()

if __name__ == '__main__':
    main()