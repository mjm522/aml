import rospy
import quaternion
import numpy as np
from aml_teleop.haptic_teleop.config import OS_TELEOP_CTRL
from aml_teleop.haptic_teleop.haptic_robot_interface import HapticRobotInterface
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_velocity_controller import OSVelocityController

class OSTeleopCtrl(HapticRobotInterface):
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
                rate : the rate of the controller in Hz (type int)
                ctrlr_type : type of the controller (either = pos, vel, or torq)
 
        """

        if config['ctrlr_type'] == 'pos':

            controller = OSPositionController(robot_interface)

        elif config['ctrlr_type'] == 'vel':

            controller = OSVelocityController(robot_interface)

        elif config['ctrlr_type'] == 'torq':

            controller = OSTorqueController(robot_interface)

        else:

            raise Exception("Unknown controller type passed!")


        HapticRobotInterface.__init__(self, haptic_interface, robot_interface, controller, config)

        self._config = config

        if not self._haptic._calibrated:

            print "**************Going to Calibrate the Haptic device and Robot******************"
            print " Instructions "
            print "1) Hold the robot end effector position in required pose"
            print "2) Hold the haptic device end effector in required pose"

            raw_input("Calibration to be performed between these two poses, press enter when ready ...")

            robot_ee_pos, robot_ee_ori = self.get_robot_ee_pose()

            self._haptic.calibration(ee_pos=robot_ee_pos,
                                     ee_ori=robot_ee_ori)


    def get_robot_ee_pose(self):

        return self._robot._state['ee_point'], self._robot._state['ee_ori']

    def get_haptic_ee_pose(self):

        return self._haptic._state['ee_point'], self._haptic._state['ee_ori']

    def compute_cmd(self):
        
        """
        this function computes the mapping between
        the haptic master and the robot slave
        if scale_from_home is set, then a motion of the 
        haptic master from its home is mapped to motion of robot from its home

        """
        
        robot_ee_pos, robot_ee_ori = self.get_robot_ee_pose()

        if not self._haptic._device_enabled:

            return robot_ee_pos, robot_ee_ori

        calib_ee_pos, calib_ee_ori = self._haptic.get_ee_pose_calib_space()

        calib_ee_ori = np.quaternion(calib_ee_ori[3], calib_ee_ori[0], calib_ee_ori[1], calib_ee_ori[2])

        return calib_ee_pos, calib_ee_ori


    def run(self):
        """
        the main code that sends control commands to the
        robot
        """        
        rate = rospy.Rate(self._config['rate'])
        
        finished = False
        
        t = 0

        while not rospy.is_shutdown() and not finished:

            self.enable_ctrlr()

            if not self._haptic._calibrated:

                print "Calibrate the Haptic device and robot ..."

                continue

            goal_ee_pos, goal_ee_ori = self.compute_cmd()

            print "Sending goal ",t, " goal_os_pos:", np.round(goal_ee_pos.ravel(), 2)

            if np.any(np.isnan(goal_ee_pos)) or np.any(np.isnan(goal_ee_ori)):
                
                print "Goal", t, "is NaN, that is not good, we will skip it!"
            
            else:
                # Setting new goal"
                
                self._ctrlr.set_goal(goal_pos=goal_ee_pos, 
                                     goal_ori=goal_ee_ori, 
                                     orientation_ctrl = True)
                
                print "Waiting..."
                
                lin_error, ang_error, success, time_elapsed  = self._ctrlr.wait_until_goal_reached(timeout=5.0)
                
                # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

            t = (t+1)

            rate.sleep()
        
        self._ctrlr.wait_until_goal_reached(timeout=5.0)
        
        self._ctrlr.set_active(False)

