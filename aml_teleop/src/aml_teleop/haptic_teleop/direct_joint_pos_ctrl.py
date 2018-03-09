import rospy
import numpy as np
from aml_teleop.haptic_teleop.config import DIRECT_JOINT_CTRL
from aml_teleop.haptic_teleop.haptic_robot_interface import HapticRobotInterface
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController


class DirectJointPosCtrl(HapticRobotInterface):
    """
    This class creates a interface between a haptic device and a robot
    
    """

    def __init__(self, haptic_interface, robot_interface, config):
        
        """
         constructor of the class
         Args:
         haptic_interface : interface to a haptic device this should of type aml_haptics
         robot_interface : interface to a robot this should of type aml_robot eg: baxter_robot
         config :
            Args: 
                robot_joints : the indices of robot joints that should be controlled (type: list)
                haptic_joints : corresponding haptic joints (type : list)
                scale_from_home : if the home of haptic device equals home of robot (type : bool)
                robot_home : if scale_from_home is set, then the home position of robot must be passed in (type : np.array)
                rate : the rate of the controller in Hz (type int)
 
        """

        HapticRobotInterface.__init__(self, haptic_interface, robot_interface, config)

        self._ctrlr = JSPositionController(robot_interface)

        self._robot = robot_interface

        self._haptic = haptic_interface

        self._config = config

        self._scale_from_home = self._config['scale_from_home']

        self._robot_home = self._config['robot_home']

        self._hap_jnts = self._config['haptic_joints']

        self._rbt_jnts = self._config['robot_joints']

        assert len(self._hap_jnts) == len(self._rbt_jnts)

        self._ctrlr.set_active(True)


    def compute_cmd(self):
        """
        this function computes the mapping between
        the haptic master and the robot slave
        if scale_from_home is set, then a motion of the 
        haptic master from its home is mapped to motion of robot from its home

        if scale_from_home is not set, then it does a linear mapping between
        the joints

        in case the robot has more joints than the haptic master
        the corresponding non-used joints will be set the the home joint position
        """

        hap_limits = self._haptic._jnt_limits

        rbt_limits = self._robot._jnt_limits

        hap_mean =  self._haptic._jnt_home

        cmd = self._robot_home.copy()

        if self._haptic._state['position'] is None:

            return

        get_curr_js_haptic = self._haptic._state['position']

        scale = np.zeros(len(self._hap_jnts))

        if self._scale_from_home:

            for k in range(len(self._hap_jnts)):

                scale[k] = (get_curr_js_haptic[k] - hap_mean[k])/(hap_limits[k]['upper'] - hap_limits[k]['lower'])

            scale[scale > 0.5]  = 0.5

            scale[scale < -0.5] = -0.5

        else:

            for k in range(len(self._hap_jnts)):

                scale[k] = (get_curr_js_haptic[k] - hap_limits[k]['lower'])/(hap_limits[k]['upper'] - hap_limits[k]['lower'])

            scale[scale > 1.] = 1.

            scale[scale < 0.] = 0.


        if self._scale_from_home:

            for i, j in zip(self._hap_jnts, self._rbt_jnts):

                cmd[j] += scale[i] * (rbt_limits[j]['upper'] - rbt_limits[j]['lower'])

        else:

            for i, j in zip(self._hap_jnts, self._rbt_jnts):

                cmd[j] = rbt_limits[j]['lower'] + scale[i] * (rbt_limits[j]['upper'] - rbt_limits[j]['lower'])

        return cmd



    def run(self):
        """
        the main code that sends control commands to the
        robot
        """        
        rate = rospy.Rate(self._config['rate'])
        
        finished = False
        
        t = 0

        while not rospy.is_shutdown() and not finished:

            goal_js_pos = self.compute_cmd()

            goal_js_vel = self._robot.get_state()['velocity']

            goal_js_acc = np.zeros_like(goal_js_pos)

            print "Sending goal ",t, " goal_js_pos:", np.round(goal_js_pos.ravel(), 2)

            if np.any(np.isnan(goal_js_pos)) or np.any(np.isnan(goal_js_vel)) or np.any(np.isnan(goal_js_acc)):
                
                print "Goal", t, "is NaN, that is not good, we will skip it!"
            
            else:
                # Setting new goal"
                
                self._ctrlr.set_goal(goal_js_pos=goal_js_pos, 
                               goal_js_vel=goal_js_vel, 
                               goal_js_acc=goal_js_acc)
                
                print "Waiting..."
                
                js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5.0)
                
                # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

            t = (t+1)

            rate.sleep()
        
        self._ctrlr.wait_until_goal_reached(timeout=5.0)
        
        self._ctrlr.set_active(False)

