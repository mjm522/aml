import rospy
import quaternion
import numpy as np
from aml_teleop.keyboard_teleop.keyboard_robot_interface import KeyboardRobotInterface
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_velocity_controller import OSVelocityController

class OSTeleopCtrl(KeyboardRobotInterface):
    """
    This class creates a interface between a haptic device and a robot
    
    """
    def __init__(self, robot_interface, config):
        
        """
         constructor of the class
         Args:
         robot_interface : interface to a robot this should of type aml_robot eg: baxter_robot
         config :
            Args: 
                rate : the rate of the controller in Hz (type int)
                ctrlr_type : type of the controller (either = pos, vel, or torq)
                robot_max_speed : speed limit
                robot_min_speed : speed limit
                ori_speed_ratio : ratio of ori change rate wrt translation
                custom_controls : for custom keyboard commands, provide dict{'key': (func_handle, [<args>], "description"}
 
        """
        
        if config['ctrlr_type'] == 'pos':

            self._ctrl_type = OSPositionController

        elif config['ctrlr_type'] == 'vel':

            self._ctrl_type = OSTorqueController

        elif config['ctrlr_type'] == 'torq':

            self._ctrl_type = OSTorqueController

        else:

            raise Exception("Unknown controller type passed!")

        controller = self._ctrl_type(robot_interface)

        KeyboardRobotInterface.__init__(self, robot_interface, controller, config)

        self._create_bindings()

        self._speed = 0.01
        self._ori_speed_ratio = self._config['ori_speed_ratio']

    def compute_cmd(self):

        pass


    def _create_bindings(self):

        """
         creates the keyboard bindings. Also adds the custom commands if provided in config file
 
        """

        self._bindings = {
                        # ----- translation
                        'w': (self._move_ee, ['x', 1.0], "move along X"),
                        's': (self._move_ee, ['x', - 1.0], "move along (-X)"),
                        'q': (self._move_ee, ['z', 1.0], "move along Z"),
                        'e': (self._move_ee, ['z', - 1.0], "move along (-Z)"),
                        'a': (self._move_ee, ['y', 1.0], "move along Y"),
                        'd': (self._move_ee, ['y', - 1.0], "move along (-Y)"),

                        # ----- rotation
                        '6': (self._turn_ee, ['x', 1.0], "turn about X"),
                        '4': (self._turn_ee, ['x', - 1.0], "turn about (-X)"),
                        '7': (self._turn_ee, ['z', 1.0], "turn about Z"),
                        '9': (self._turn_ee, ['z', - 1.0], "turn about (-Z)"),
                        '8': (self._turn_ee, ['y', 1.0], "turn about Y"),
                        '5': (self._turn_ee, ['y', - 1.0], "turn about (-Y)"),

                        # ----- speed scale
                        '+': (self._change_speed, [0.01], "increase spead by 0.01"),
                        '-': (self._change_speed, [-0.01], "decrease spead by 0.01"),
                     }

        if 'custom_controls' in self._config and  self._config['custom_controls'] is not None:
            print "YES!!!!"
            for key, value in self._config['custom_controls'].iteritems():
                self._bindings.setdefault(key, []).extend(value)

    def _change_speed(self, inc):

        """
            function to change the rate of change of end-effector pose per keystroke
        """

        self._speed += inc
        
        if self._speed >= self._config['robot_max_speed']:
            print "Allowed Speed Limit reached!"
            self._speed = self._config['robot_max_speed']

        elif self._speed < 0:
            print "Speed Cannot be reduced further"
            self._speed = 0

        print "New Speed: %f"%self._speed


    def _move_ee(self, axis, speed):

        """
            moves the end-effector in the task space
            Args:
                axis : string (x,y,z) : axis along which to move the end-effector
                speed : +1 or -1 : for direction
        """

        curr_ee_pos, curr_ee_ori = self.get_robot_ee_pose()

        speed*= self._speed

        if axis == 'y':

            pos_des = curr_ee_pos + np.asarray([0,speed,0])

        elif axis == 'z':

            pos_des = curr_ee_pos + np.asarray([0,0,speed])

        elif axis == 'x':

            pos_des = curr_ee_pos + np.asarray([speed,0,0])

        if np.any(np.isnan(pos_des)) or np.any(np.isnan(curr_ee_ori)):
                
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        
        else:

            self._ctrlr.set_goal(goal_pos=pos_des, 
                               goal_ori=curr_ee_ori, 
                               orientation_ctrl = True)
            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)

        print "current pose", curr_ee_pos


    def _turn_ee(self, axis, speed):

        """
            turns the end-effector in the task space
            Args:
                axis : string (x,y,z) : axis along which to rotate the end-effector
                speed : +1 or -1 : for direction
        """

        curr_ee_pos, curr_ee_ori = self.get_robot_ee_pose()

        curr_ee_ori = quaternion.as_euler_angles(curr_ee_ori)
        # print curr_ee_pos, curr_ee_ori
        # print self._robot.state()['ee_ori']

        # print self._robot.inv_kin(curr_ee_pos, curr_ee_ori)
        speed*= (self._speed*self._ori_speed_ratio)

        if axis == 'x':

            ori_des = curr_ee_ori + np.asarray([0,speed,0])

        elif axis == 'z':

            ori_des = curr_ee_ori + np.asarray([0,0,speed])

        elif axis == 'y':

            ori_des = curr_ee_ori + np.asarray([speed,0,0])

        if np.any(np.isnan(curr_ee_pos)) or np.any(np.isnan(ori_des)):
                
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        
        else:

            ori_des = quaternion.from_euler_angles(ori_des)

            self._ctrlr.set_goal(goal_pos=curr_ee_pos, 
                               goal_ori=ori_des, 
                               orientation_ctrl = True)
            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)

        print "current pose", curr_ee_pos


    def run(self):

        """
            The main function which runs the teleop code
        """

        self.enable_ctrlr()

        print("\n\n\tControlling End Effector of %s. Press ? for help, Esc to quit."%type(self._robot).__name__)
        self._run()

        self.disable_ctrlr()
        print("\nDone.")



