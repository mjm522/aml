import rospy # Needed for nodes, rate, sleep, publish, and subscribe.
import tf
from tf import TransformListener # Needed for listening to and transforming robot state information.
from aml_ctrl.classical_controllers  import MinJerkController
from aml_robot.baxter_robot import BaxterArm, BaxterButtonStatus
from baxter_interface import CHECK_VERSION
import baxter_interface
import numpy as np
import quaternion
import sys
import copy
import time
import threading # Used for time locks to synchronize position data.
from threading import Timer
from os.path import dirname, abspath
#get the parent folder
data_folder_path = dirname(dirname(abspath(__file__))) + '/data/'

class LfD():

    def __init__(self):
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        rs.enable()
        self.left_arm 	= BaxterArm('left') 
        self.right_arm  = BaxterArm('right')
        self.controller = MinJerkController(extern_call=True, 
                                            trial_arm=self.left_arm, 
                                            aux_arm=self.right_arm)
        #this will be rate at which data will be read from the arm
        self.sampling_rate = 100
        self.left_arm.set_sampling_rate(sampling_rate=self.sampling_rate):
        self.stale_observation = False
        #time at which the baxter data should be called
        self.sample_period = 0.5

    def timer(self):
        self.stale_observation = False

    def arm_status(self, limb_idx=1):
        
        if limb_idx == 0:
            arm = self.left_arm
            q_mean = np.array([-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50])
        elif limb_idx == 1:
            arm = self.right_arm
            q_mean = np.array([0.08, -1.0,  1.19, 1.94, -0.67, 1.03,  0.50])
        else:
            print "Unknown limb index"
            raise ValueError

        arm_state      = arm._state
        # calculate position of the end-effector
        ee_xyz, ee_ori = arm.get_ee_pose()
        arm_state['ee_pos'] = ee_xyz
        arm_state['ee_ori'] = quaternion.as_float_array(ee_ori)[0]
        #this is for the torque controller
        arm_state['jnt_start'] = q_mean

        return arm_state

    def load_demo_data(self, demo_idx=0, modified=False):
        # Load
        try:
            if not modified:
                demo_data = np.load(data_folder_path+'demo_data.npy')
            else:
                demo_data = np.load('demo_data_modified.npy')
        
        except Exception as e:
            print "Caliberation file cannot be loaded"
            raise ValueError

        return demo_data

    def plot_demo_data(self):
        demo_data = self.load_demo_data()
        ee_list = []
        for arm_data in demo_data:
            ee_list.append(arm_data['ee_pos'])
        ee_list =  np.asarray(ee_list).squeeze()
        ee_vel_list = np.diff(ee_list, axis=0)
        #print ee_list
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
        ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
        ax.plot(ee_list[:,0],ee_list[:,1],ee_list[:,2])
        ax.plot(ee_vel_list[:,0],ee_vel_list[:,1],ee_vel_list[:,2], color='m')
        print ee_vel_list
        plt.show()


    def store_torque_cmds(self):
        demo_data = self.load_demo_data()
        arm_demo = []
        torque_mags = []
        for indx in range(0, len(demo_data)-1):
            arm_data_curr = demo_data[indx]
            tmp_data = copy.deepcopy(arm_data_curr)
            arm_data_nxt = demo_data[indx+1]
            q_curr = arm_data_curr['ee_ori']
            q_nxt  = arm_data_nxt['ee_ori']

            #orientation need to converted to be used by minjerkcontroller
            tmp_data['ee_ori'] = np.quaternion(q_curr[0],q_curr[1],q_curr[2],q_curr[3])

            cmd = self.controller.osc_torque_cmd_2(arm_data=tmp_data, 
                                                   goal_pos=arm_data_nxt['ee_pos'], 
                                                   goal_ori=np.quaternion(q_nxt[0],q_nxt[1],q_nxt[2],q_nxt[3]),
                                                   orientation_ctrl=True)
            
            
            #cmd = 0.1*(arm_data_curr['effort']-arm_data_curr['gravity_comp'])

            #print "goal diff \n", arm_data_nxt['ee_pos'] - arm_data_curr['ee_pos']
            arm_data_curr['torque'] = cmd

            torque_mags.append(np.linalg.norm(cmd))
            arm_demo.append(arm_data_curr)
            

        import matplotlib.pyplot as plt
        plt.plot(torque_mags)
        plt.show()
        np.save(data_folder_path+'demo_data_modified.npy', arm_demo)

    def check_demo_data(self, demo_idx=0, limb_idx=1):
        
        if limb_idx == 0:
            arm = self.left_arm
        elif limb_idx == 1:
            arm = self.right_arm
        else:
            print "Unknown limb index"
            raise ValueError

        demo_data = self.load_demo_data(modified=True)
        #set the arm to initial joint demo position
        arm.move_to_joint_position(demo_data[0]['position'])
        rate = rospy.timer.Rate(self.sampling_rate)
        for arm_data in demo_data:
            #apply computed torques
            #arm.exec_torque_cmd(arm_data['torque'])
            rate.sleep()
            arm.move_to_joint_position(arm_data['position'])

            arm.exec_torque_cmd(arm_data['torque'])
            rate.sleep()
            #arm.move_to_joint_position(arm_data['position'])

    def save_demo_data(self, limb_idx=0):
        btn = BaxterButtonStatus()
        
        demo_start_flag = False
        arm_demo = []
        rate = rospy.timer.Rate(self.sampling_rate)
        #btn.right_dash_btn_state will be false initially
        while True:
            if btn.right_dash_btn_state is True:
                arm_demo.append(self.arm_status())
                rate.sleep()
                demo_start_flag = True

            if demo_start_flag and (not btn.right_dash_btn_state):
                break

        arm_demo[0]['limb_idx'] = limb_idx
        arm_demo[0]['sampling_rate'] = rate
        np.save(data_folder_path+'demo_data.npy', arm_demo)


def main(args):
    lfd = LfD()
    
    if 'save' in args:
        lfd.save_demo_data(limb_idx=args['limb_idx'])
    
    if 'plot' in args:
        lfd.plot_demo_data()

    if 'torque' in args:
        lfd.store_torque_cmds()

    if 'check' in args:
        lfd.check_demo_data()


if __name__ == '__main__':
    rospy.init_node('lfd_node')
    cmdargs = str(sys.argv)
    main(cmdargs)
