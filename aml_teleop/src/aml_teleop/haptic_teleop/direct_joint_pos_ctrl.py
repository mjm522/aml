import numpy as np
from aml_teleop.haptic_teleop.config import DIRECT_JOINT_CTRL
from aml_teleop.haptic_teleop.haptic_robot_interface import HapticRobotInterface
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController


class DirectJointPosCtrl(HapticRobotInterface):

    def __init__(self, haptic_interface, robot_interface, config):

        HapticRobotInterface.__init__(self, haptic_interface, robot_interface, config)

        self._ctrlr = JSPositionController(robot_interface)

        self._robot = robot_interface

        self._haptic = haptic_interface

        self._config = config

        self._hap_jnts = self._config['haptic_joints']

        self._rbt_jnts = self._config['robot_joints']

        self._ctrlr.set_active(True)


    def compute_cmd(self):

        hap_limits = self._haptic._jnt_limits

        rbt_limits = self._robot._jnt_limits

        get_curr_js_haptic = self._haptic._state['position']

        ratio = np.zeros(len(self._hap_jnts))

        for k in range(len(self._hap_jnts)):

            ratio[k] = get_curr_js_haptic[k]/(hap_limits[k]['upper'] - hap_limits[k]['lower'])

        cmd = np.zeros(len(rbt_limits))

        for i,j in zip(self._hap_jnts, self._rbt_jnts):

            cmd[j] = (rbt_limits[j]['upper'] + rbt_limits[j]['lower'])*ratio[i]


        return cmd



    def run(self):
        
        rate = rospy.Rate(100)
        
        finished = False
        
        t = 0

        while not rospy.is_shutdown() and not finished:

            goal_js_pos = self.compute_cmd()

            print goal_js_pos
        #     goal_js_vel = self._robot.get_state()['velocity']
        #     goal_js_acc = np.zeros_like(goal_js_pos)

        #     print "Sending goal ",t, " goal_js_pos:", np.round(goal_js_pos.ravel(), 2)

        #     if np.any(np.isnan(goal_js_pos)) or np.any(np.isnan(goal_js_vel)) or np.any(np.isnan(goal_js_acc)):
                
        #         print "Goal", t, "is NaN, that is not good, we will skip it!"
            
        #     else:
        #         # Setting new goal"
                
        #         self._ctrlr.set_goal(goal_js_pos=goal_js_pos, 
        #                        goal_js_vel=goal_js_vel, 
        #                        goal_js_acc=goal_js_acc)
                
        #         print "Waiting..."
                
        #         js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5.0)
                
        #         # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        #     t = (t+1)

        #     rate.sleep()
        
        # self._ctrlr.wait_until_goal_reached(timeout=5.0)
        
        # self._ctrlr.set_active(False)

