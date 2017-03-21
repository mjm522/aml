import os
import copy
import rospy
import argparse
import numpy as np
from aml_robot.baxter_robot import BaxterArm
from aml_data_collec_utils.core.data_recorder import DataRecorder
from aml_lfd.utilities.store_demonstration import StoreDemonstration
from aml_io.io_tools import save_data, load_data, get_aml_package_path
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

reset_jnt_positions = {}

storage_path = os.environ['AML_DATA'] + '/aml_data_collec_utils/reset_box_trajectories/'

def save_jnt_position(baxter_arm):
    
    global reset_jnt_positions

    while True:

        choice = raw_input("Enter y to record this joint state n for not to! press e to end: \t")

        if choice == 'y':
            key_name = raw_input("Enter key to be used to store this sample:\t")
            reset_jnt_positions[key_name] = baxter_arm._state['position']

        elif choice == 'n':
            print "\n Not saving this joint position!"

        elif choice == 'e':

            write_file()
            break

        else:
            continue

def go_to_default_pose(baxter_arm):
    global storage_path
    filename = storage_path+'reset_joint_positions.pkl'

    new_record = False

    if os.path.exists(filename):

        reset_joint_pos=load_data(filename)

        if reset_joint_pos.has_key('default_pose'):
            baxter_arm.move_to_joint_position(reset_joint_pos['default_pose'])
        else:
            print "File exists but the 'key: default_pose' cannot be found"
            new_record = True
    else:

        new_record = True

    if new_record:

        choice = raw_input('Do you want to record a default pose now?(y/n)')

        if choice == 'y':
            save_jnt_position(baxter_arm)
        else:
            return

def write_file():
    global reset_jnt_positions
    global storage_path

    append_to_file = False

    filename = storage_path+'reset_joint_positions.pkl'

    if not bool(reset_jnt_positions):
        print "\n Nothing to save!"
        return

    if os.path.exists(filename):
        choice=raw_input('A reset file already exits, do you want to add to that existing file? (y/n) \t')
        if choice == 'y':
            append_to_file = True

    save_data(reset_jnt_positions, filename, append_to_file)


def check_jnt_position(baxter_arm):
    global storage_path

    reset_jnt_positions = load_data(storage_path+'reset_joint_positions.pkl')

    for key in reset_jnt_positions.keys():
        print "Going to %s"%(key,)
        baxter_arm.move_to_joint_position(reset_jnt_positions[key])
        choice=raw_input('Do you want to play next pose? (y/n)')
        if choice=='n':
            break

def get_sweep_goto_path(baxter_arm, side_name):

    goto_path     = storage_path+'right_arm_goto_'+side_name+'_01.pkl'
    sweep_path    = storage_path+'right_arm_sweep_'+side_name+'_01.pkl'
    kwargs_goto   = {'path_to_demo':goto_path, 'limb_name':baxter_arm._limb}
    kwargs_sweep  = {'path_to_demo':sweep_path,'limb_name':baxter_arm._limb}
    goto_js_traj  = JSTrajGenerator(load_from_demo=True, **kwargs_goto)
    sweep_js_traj = JSTrajGenerator(load_from_demo=True, **kwargs_sweep)

    return goto_js_traj, sweep_js_traj

def check_sweep_demo(baxter_arm):
    global storage_path
    reset_jnt_positions = load_data(storage_path+'reset_joint_positions.pkl')
    
    choice=raw_input('Which sweep do you want to check?(front/right/left/back) \t')

    if choice=='front':
        go_to_default_pose(baxter_arm)
        goto_js_traj, sweep_js_traj = get_sweep_goto_path(baxter_arm, 'front')

    elif choice=='right':
        go_to_default_pose(baxter_arm)
        goto_js_traj, sweep_js_traj = get_sweep_goto_path(baxter_arm, 'right')

    elif choice=='left':
        go_to_default_pose(baxter_arm)
        goto_js_traj, sweep_js_traj = get_sweep_goto_path(baxter_arm, 'left')

    elif choice=='back':
        go_to_default_pose(baxter_arm)
        goto_js_traj, sweep_js_traj = get_sweep_goto_path(baxter_arm, 'back')

    else:
        print "Incorrect choice"
        return

    goto_js_traj.generate_traj()
    sweep_js_traj.generate_traj()

    rate = rospy.Rate(10)

    for pos_cmd in goto_js_traj._traj['pos_traj']:
        baxter_arm.exec_position_cmd(pos_cmd)
        rate.sleep()

    print "Reached sweep pose"

    for pos_cmd in sweep_js_traj._traj['pos_traj']:
        baxter_arm.exec_position_cmd(pos_cmd)
        rate.sleep()

    print "Finished sweep"

    print "Going back"

    for pos_cmd in sweep_js_traj._traj['pos_traj'][::-1]:
        baxter_arm.exec_position_cmd(pos_cmd)
        rate.sleep()

    for pos_cmd in goto_js_traj._traj['pos_traj'][::-1]:
        baxter_arm.exec_position_cmd(pos_cmd)
        rate.sleep()

    print "I should have reached the default pose now"


def fsm_reset(baxter_arm, order_of_sweep, rate=15):

    reset_traj = []

    for side_name in order_of_sweep:

         goto_js_traj, sweep_js_traj = get_sweep_goto_path(baxter_arm, side_name)

         goto_js_traj.generate_traj()
         sweep_js_traj.generate_traj()

         tmp_goto   = goto_js_traj._traj['pos_traj'].tolist()
         tmp_sweep  = sweep_js_traj._traj['pos_traj'].tolist()
         tmp_goto_  = copy.deepcopy(tmp_goto)
         tmp_sweep_ = copy.deepcopy(tmp_sweep)
         tmp_goto_.reverse()
         tmp_sweep_.reverse()

         reset_traj = reset_traj + tmp_goto + tmp_sweep + tmp_sweep_ + tmp_goto_

    rate= rospy.Rate(rate)
    for pos_cmd in reset_traj:
        baxter_arm.exec_position_cmd(pos_cmd)
        rate.sleep()


def record_joint_demo(baxter_arm):
    
    global storage_path

    choice = raw_input('Do you want to record from default pose?(y/n) \t')

    if choice=='y':
        go_to_default_pose(baxter_arm)

    data_name_prefix = raw_input('Enter name of the demonstration: \t')

    lfd = StoreDemonstration(robot_interface=baxter_arm,    demo_idx=0, 
                             data_folder_path=storage_path, data_name_prefix='arm_'+data_name_prefix)

    lfd.save_demo_data()

    if lfd._finish_demo:

        lfd.save_now()



def main():

    parser = argparse.ArgumentParser(description='Data collection for push manipulation')
    
    parser.add_argument('-i', '--input', type=str, help='-i save  : store reset positions \n\
                                                         -i check : cehck the stored positions \n\
                                                         -i demo  : store a demo \n\
                                                         -i demo_check : check a stored demo \n\
                                                         -i fsm : check fsm reset')
    
    args = parser.parse_args()

    rospy.init_node('baxter_reset_box', anonymous=True)

    arm = BaxterArm('right')

    while not rospy.is_shutdown():

        if args.input=='save':
            save_jnt_position(arm)
        
        elif args.input=='check':
            check_jnt_position(arm)
            
        elif args.input=='demo':
            record_joint_demo(arm)

        elif args.input=="demo_check":
            check_sweep_demo(arm)

        elif args.input=='fsm':

            order_of_sweep = ['left', 'back', 'front', 'right']
            fsm_reset(arm, order_of_sweep)

        else:
            print "For help, press -h"

        choice=raw_input('Do you want to continue...(y/n) \t')
        
        if choice=='n':
                break
        else:
            args.input=raw_input('Do you want to do anything else? (save/check/demo/demo_check) \t')


if __name__ == '__main__':
    main()


