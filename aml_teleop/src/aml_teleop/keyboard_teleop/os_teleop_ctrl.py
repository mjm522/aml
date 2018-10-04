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
 
        """

        if config['ctrlr_type'] == 'pos':

            controller = OSPositionController(robot_interface)

        elif config['ctrlr_type'] == 'vel':

            controller = OSVelocityController(robot_interface)

        elif config['ctrlr_type'] == 'torq':

            controller = OSTorqueController(robot_interface)

        else:

            raise Exception("Unknown controller type passed!")


        KeyboardRobotInterface.__init__(self, robot_interface, controller, config)

        self._create_bindings()

        self._speed = 0.01
        self._ori_speed_ratio = 0.5


    def _create_bindings(self):

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

        if 'custom_commands' in self._config and  self._config['custom_commands'] is not None:
            for key, value in self._config['custom_commands'].iteritems():
                self._bindings.setdefault(key, []).extend(value)

    def compute_cmd(self):
        
        pass

    def _change_speed(self, inc):

        if 0.0 < self._speed < self._config['robot_max_speed']:
            self._speed += inc
            print "New Speed: %f"%self._speed
        else:
            print "Allowed Speed Limit reached!"


    def _move_ee(self, axis, speed):

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
        # self._robot.set_joint_state(cmd)
        # print new_pos, new_ori

    def _turn_ee(self, axis, speed):

        curr_ee_pos, curr_ee_ori = self.get_robot_ee_pose()

        curr_ee_ori = quaternion.as_euler_angles(curr_ee_ori)
        # print curr_ee_pos, curr_ee_ori
        # print self._robot.state()['ee_ori']

        # print self._robot.inv_kin(curr_ee_pos, curr_ee_ori)
        speed*= (self._speed*self._ori_speed_ratio)

        if axis == 'y':

            ori_des = curr_ee_ori + np.asarray([0,speed,0])

        elif axis == 'z':

            ori_des = curr_ee_ori + np.asarray([0,0,speed])

        elif axis == 'x':

            ori_des = curr_ee_ori + np.asarray([speed,0,0])

        if np.any(np.isnan(curr_ee_pos)) or np.any(np.isnan(ori_des)):
                
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        
        else:

            ori_des = quaternion.from_euler_angles(ori_des)

            self._ctrlr.set_goal(goal_pos=curr_ee_pos, 
                               goal_ori=ori_des, 
                               orientation_ctrl = True)
            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)


    def run(self):

        print("Controlling End Effector. Press ? for help, Esc to quit.")
        self._run()

        print("Done.")



