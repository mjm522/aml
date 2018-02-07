import roslib
roslib.load_manifest('aml_robot')

import rospy

import numpy as np
import quaternion

from std_msgs.msg import (
    UInt16,
)

from aml_math.quaternion_utils import compute_omg

from aml_robot.bullet.bullet_robot import BulletRobot
from aml_robot.sawyer_kinematics import sawyer_kinematics
from aml_robot.sawyer_ik import IKSawyer

# from aml_visual_tools.load_aml_logo import load_aml_logo

class BulletSawyerArm(BulletRobot):

    def __init__(self, robot_id, limb = "right", on_state_callback=None):

        #Load aml_logo
        # load_aml_logo("/robot/head_display")
        # getNumJoints(bodyUniqueId)

        BulletRobot.__init__(self,robot_id, ee_link_idx = 16) # hardcoded from the sawyer urdf

        self._ready = False

        self._limb = limb

        self._configure(limb,on_state_callback)
        
        self._ready = True

        #number of joints
        self._nq = 7
        #number of control commads
        self._nu = 7

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


        self._movable_joints = self._joint_idx

        print "movable joints = ", self._movable_joints, len(self._movable_joints),

        self._ee_pos_old, self._ee_ori_old = self.get_ee_pose()
        self._time_now_old = self.get_time_in_seconds()

    def _configure_cuff(self):
        print "NOT IMPLEMENTED"

    def _configure_gripper(self):
        print "NOT IMPLEMENTED"

    def _open_action(self, value):
        print "NOT IMPLEMENTED"

    def _close_action(self, value):
        print "NOT IMPLEMENTED"

    def _light_action(self, value):
        print "NOT IMPLEMENTED"

    def _set_lights(self, color, value):
        print "NOT IMPLEMENTED"

    def cuff_cb(self, value):
        print "NOT IMPLEMENTED"

#     #this function returns self._cuff_state to be true
#     #when arm is moved by a demonstrator, the moment arm stops 
#     #moving, the status returns to false
#     #initial value of the cuff is None, it is made False by pressing the
#     #cuff button once
    @property
    def get_lfd_status(self):
        print "NOT IMPLEMENTED"

    def set_sampling_rate(self, sampling_rate=100):
        self._pub_rate.publish(sampling_rate)

    def tuck_arm(self):
        print "NOT IMPLEMENTED"

    def untuck_arm(self):
        # for i in range(len(self._movable_joints)):
        #     self.set_jnt_state(self._movable_joints[i],self._untuck[i])
        self.move_using_pos_control(self._movable_joints, self._untuck)
        print "----------------------Untucking Complete"

    def _configure(self, limb, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        # Parent constructor
        # intera_interface.Limb.__init__(self, limb)

        # bullet_interface(self,limb)

        # self._kinematics = sawyer_kinematics(self)

        # self._ik_sawyer = IKSawyer(limb=self)

        # self._ik_sawyer.configure_ik_service()

        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)

        # self._gravity_comp = rospy.Subscriber('robot/limb/' + limb + '/gravity_compensation_torques', SEAJointState, self._gravity_comp_callback)

        #gravity + feed forward torques
        self._h = [0. for _ in range(7)]

        self.set_sampling_rate()

        # self.set_command_timeout(0.2)

        self._camera = None
        self._gripper = None
        self._cuff = None

        # self._configure_cuff()
        # self._configure_gripper()


    def _gravity_comp_callback(self, msg):
        self._h = msg.gravity_model_effort
#         # print "commanded_effort \n",   msg.commanded_effort
#         # print "commanded_velocity \n", msg.commanded_velocity
#         # print "commanded_position \n", msg.commanded_position
#         # print "actual_position \n", msg.actual_position
#         # print "actual_velocity \n", msg.actual_velocity
#         # print "actual_effort \n", msg.actual_effort
#         # print "gravity_model_effort \n", msg.gravity_model_effort
#         # print "hysteresis_model_effort \n", msg.hysteresis_model_effort
#         # print "crosstalk_model_effort \n", msg.crosstalk_model_effort
#         # print "difference effort \n", np.array(msg.commanded_effort) - np.array(msg.actual_effort)
#         # print "difference velocity \n", np.array(msg.commanded_velocity) - np.array(msg.actual_velocity)
#         # print "difference position \n", np.array(msg.commanded_position) - np.array(msg.actual_position)


    def get_gripper_state(self):
        print 'NOT IMPLEMENTED'


    def set_gripper_speed(self, speed):
        print 'NOT IMPLEMENTED'

    def set_arm_speed(self,speed):
        print 'NOT IMPLEMENTED'
        

    def _on_joint_states(self, msg):
        print 'NOT IMPLEMENTED'

    def get_end_effector_link_name(self):
        print 'NOT IMPLEMENTED'
    
    def get_base_link_name(self):
        print 'NOT IMPLEMENTED'

    def exec_gripper_cmd(self, pos, force = None):
        print 'NOT IMPLEMENTED'

    def exec_gripper_cmd_delta(self, pos_delta, force_delta = None):
        print 'NOT IMPLEMENTED'

    def exec_position_cmd(self, cmd):
        print 'NOT IMPLEMENTED'

    def exec_position_cmd_delta(self,cmd):
        print 'NOT IMPLEMENTED'

    def move_to_joint_pos_delta(self,cmd):
        curr_q = self._state['position']
        # joint_names = self.joint_names()
        vals = [i+j for i,j in zip(curr_q,cmd)]

        self.move_to_joint_pos(vals)

    def move_to_joint_pos(self,cmd):

        self.move_using_pos_control(self._movable_joints, cmd)

    def exec_velocity_cmd(self,cmd):
                
        self.set_joint_velocities(cmd)

    def exec_torque_cmd(self,cmd):

        self.set_joint_torques(cmd)

#     def move_to_joint_position(self, joint_angles):
#         self.move_to_joint_positions(dict(zip(self.joint_names(),joint_angles)))


    def get_ee_velocity(self, from_bullet=True):
        
        if from_bullet:

            return self.get_ee_velocity_from_bullet()

        else:

            #this is a simple finite difference based velocity computation
            #please note that this might produce a bug since self._goal_ori_old gets 
            #updated only if get_ee_vel is called. 
            #TODO : to update in get_ee_pose or find a better way to compute velocity

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
        
        print 'NOT IMPLEMENTED'
        # if joint_angles is None:
            
        #     argument = None
        
        # else:
            
        #     argument = dict(zip(self.joint_names(),joint_angles))
        
        # #combine the names and joint angles to a dictionary, that only is accepted by kdl
        # pose = np.array(self._kinematics.forward_position_kinematics(argument))
        # position = pose[0:3][:,None] #senting as  column vector
        
        # w = pose[6]; x = pose[3]; y = pose[4]; z = pose[5] #quarternions
        
        # #formula for converting quarternion to rotation matrix

        # rotation = np.array([[1.-2.*(y**2+z**2),    2.*(x*y-z*w),           2.*(x*z+y*w)],\
        #                      [2.*(x*y+z*w),         1.-2.*(x**2+z**2),      2.*(y*z-x*w)],\
        #                      [2.*(x*z-y*w),         2.*(y*z+x*w),           1.-2.*(x**2+y**2)]])
        
        # return position, rotation

    def get_cartesian_vel_from_joints(self, joint_angles=None):

        print 'NOT IMPLEMENTED'
        
#         if joint_angles is None:
            
#             argument = None
        
#         else:
            
#             argument = dict(zip(self.joint_names(),joint_angles))
        
#         #combine the names and joint angles to a dictionary, that only is accepted by kdl
#         return np.array(self._kinematics.forward_velocity_kinematics(argument))[0:3] #only position

    def get_jacobian_from_joints(self, joint_angles=None):
        print "NOT IMPLEMENTED: cannot calculate jacobians for the movable joints alone"


    def get_arm_inertia(self, joint_angles=None):
        
        return self.get_inertia_matrix()

    def ik(self, pos, ori=None):
        print 'NOT IMPLEMENTED'

def main():
    rospy.init_node("sawyer_arm_example")

    arm = BulletSawyerArm('right')

    rospy.spin()

if __name__ == '__main__':
    main()