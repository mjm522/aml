import rospy
import argparse
from aml_io.io_tools import save_data, load_data, get_aml_package_path
from aml_robot.baxter_robot import BaxterArm

reset_jnt_positions = {}
filename = get_aml_package_path('aml_data_collec_utils') + '/utilities/reset_joint_positions.pkl'

def save_jnt_position():
    
    global reset_jnt_positions

    while True:

        choice = raw_input("Enter y to record this joint state n for not to! press e to end: \t")

        if choice == 'y':
            key_name = raw_input("Enter key to be used to store this sample:\t")
            reset_jnt_positions[key_name] = jnt_pos

        elif choice == 'n':
            print "\n Not saving this joint position!"

        elif choice == 'e':

            write_file()
            break

        else:
            continue


def write_file():
    global reset_jnt_positions
    global filename

    if not bool(reset_jnt_positions):
        print "\n Nothing to save!"
        return

    save_data(reset_jnt_positions, filename)

def on_shutdown():
    #this if for saving files in case keyboard interrupt happens
    write_file()


def check_jnt_position(baxter_arm):
    global filename
    reset_jnt_positions = load_data(filename)

    for key in reset_jnt_positions.keys():
        print "Going to %s"%(key,)
        baxter_arm.move_to_joint_position(reset_jnt_positions[key])



def main():

    parser = argparse.ArgumentParser(description='Data collection for push manipulation')
    
    parser.add_argument('-s', '--save', type=str, help='store reset positions')
    parser.add_argument('-c', '--check', type=str, help='check the stored samples')
    
    args = parser.parse_args()

    rospy.init_node('baxter_reset_box', anonymous=True)

    arm = BaxterArm('right')

    rospy.on_shutdown(on_shutdown)

    while not rospy.is_shutdown():

        if args.save=='save':
            save_jnt_position()
        elif args.check=='check':
            check_jnt_position(arm)
            choice=raw_input('Do you want to continue...(y/n)')
            if choice=='n':
                break
        else:
            print "For help, press -h"


if __name__ == '__main__':
    main()


