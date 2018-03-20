import os
import cv2
import rospy
import argparse
import numpy as np
import quaternion
from aml_robot.baxter_robot import BaxterArm
from aml_io.convert_tools import image2string
from aml_io.io_tools import save_data, load_data
from aml_visual_tools.visual_tools import show_image
from aml_data_collec_utils.box_object import BoxObject

test_save_location = os.environ['AML_DATA'] + '/aml_dl/mdn/test_images/'
pre_processed_data_location = os.environ['AML_DATA'] + '/aml_dl/pre_process_data_siamese/test/'

if not os.path.exists(test_save_location):
    os.makedirs(test_save_location)

class TestSetup(BoxObject):

    def __init__(self, robot_interface):
        self._box = BoxObject()
        self._robot = robot_interface

    def check_kinematic_feasibility(self, goals):
    
        success = self.goto_goals(goals)

        if not success:
            return False
        else:
            return True

    def pack_push_goals(self, push):

        goals = []
        for goal in push['poses']:
            goals.append(goal)

        push_action = push['push_action']

        goals.append({'pos': push_action, 'ori': None})

        return goals
        
    def goto_pose(self, goal_pos, goal_ori): 

        start_pos, start_ori = self._robot.ee_pose()

        if goal_ori is None:
             goal_ori = start_ori

        goal_ori = quaternion.as_float_array(goal_ori)
        success, js_pos = self._robot.inverse_kinematics(goal_pos, goal_ori)

        if success:
            self._robot.move_to_joint_position(js_pos)
        else:
            print "Couldnt find a solution"

        return success

    def goto_goals(self, goals):

        c = 0
        success = False
        for i in range(len(goals)-1):
            goal = goals[i]
            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)

            if not success:
                return False

        # Push is always the last
        goal = goals[len(goals)-1]
        print "Gonna push the box ..."
        success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)
            
        return success

    def record_test_data(self):
        pushes, pose, reset_push = self._box.get_pushes(use_random=False)
        '''
        if box if kept on the marker on table, facing baxter (according to arrow on box)
        push 0 - side d of box
        push 1 - side b 
        push 2 - side c of box
        push 3 - side a of box
        '''
        for k in range(4):
            self._robot.untuck()
            goals = self.pack_push_goals(pushes[k])
            curr_state =  self._box.get_curr_image()
            s_box_pose, s_box_pos, s_box_q = self._box.get_pose()
            test_data = {}
            if self.check_kinematic_feasibility(goals):
                
                goals.reverse()
                success = self.goto_goals(goals[1:])
                self._robot.untuck()
                if success:
                    final_state =  self._box.get_curr_image()
                    f_box_pose, f_box_pos, f_box_q = self._box.get_pose()

                    test_data['s_image']    = curr_state
                    test_data['s_box_pos']  = s_box_pos
                    test_data['s_box_q']    = s_box_q
                    test_data['s_box_pose'] = s_box_pose
                    test_data['f_image']    = final_state
                    test_data['f_box_pos']  = f_box_pos
                    test_data['f_box_q']    = f_box_q
                    test_data['f_box_pose'] = f_box_pose


                    raw_input('adjust box manually')

                    final_state =  self._box.get_curr_image()
                    f_box_pose, f_box_pos, f_box_q = self._box.get_pose()

                    test_data['fm_image']    = final_state
                    test_data['fm_box_pos']  = f_box_pos
                    test_data['fm_box_q']    = f_box_q
                    test_data['fm_box_pose'] = f_box_pose

                    save_data(test_data, filename=test_save_location+'test_data_'+str(k)+'.pkl')



                    print "Saved data!"
            else:
                print "Failed push ", k

            raw_input("Move the box to default pos, press enter to continue...")

    def load_test_data(self):
        for k in range(4):
            try:
                data = load_data(filename=test_save_location+'test_data_'+str(k)+'.pkl')
            except Exception as e:
                print 'Unable to load file, trying next one'
                continue
            show_image(data['s_image'], window_name='Start image')
            show_image(data['f_image'], window_name='Final image')
        
        
def main(args):
    rospy.init_node('get_test_images', anonymous=True)
    limb = 'left'
    arm = BaxterArm(limb)
    ts =  TestSetup(robot_interface=arm)
    if args.operation == 'rec':
        ts.record_test_data()
    elif args.operation == 'load':
        ts.load_test_data()
    else:
        print "Unknown argument \t", args.operation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data collection for push manipulation')
    parser.add_argument('-i', '--operation', type=str, help='type -i rec for recording test images, -i load for checking the files')
    args = parser.parse_args()
    if args is None:
        print "Enter arguments, -h for help"
    else:
        main(args)