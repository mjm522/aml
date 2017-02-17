import tf# Needed for listening to and transforming robot state information.
import copy
import time
import rospy # Needed for nodes, rate, sleep, publish, and subscribe.
import argparse
import numpy as np
import quaternion
from tf import TransformListener
from aml_io.io_tools import get_aml_package_path
from aml_data_collec_utils.core.data_recorder import DataRecorder


class Task():

    def __init__(self):
        pass

    def get_status(self):

        return True

    def get_effect(self):

        dummy_task = {}

        return dummy_task

class StoreDemonstration():

    def __init__(self, robot_interface, demo_idx, data_folder_path=None, data_name_prefix=None, sampling_rate=100):

        self._robot = robot_interface
        #this will be rate at which data will be read from the arm
        self._sampling_rate = sampling_rate
        
        self._robot.set_sampling_rate(sampling_rate=self._sampling_rate)

        self._demo_idx = demo_idx

        if data_folder_path is None:
            data_folder_path = get_aml_package_path('aml_lfd') + '/data/'

        if data_name_prefix is None:
            data_name_prefix = robot_interface._limb + '_demo_data'
        else:
            data_name_prefix = robot_interface._limb + '_' + data_name_prefix


        self._record_sample = DataRecorder(robot_interface=robot_interface, 
                                           task_interface=Task(),
                                           data_folder_path=data_folder_path,
                                           data_name_prefix=data_name_prefix,
                                           num_samples_per_file=1, 
                                           record_rate = 30)

        self._start_record = False

        self._finish_demo  = False

    def save_demo_data(self):

        while (not self._finish_demo) and (not rospy.is_shutdown()):

            print "Robot cuff state \t", self._robot.get_lfd_status

            if self._robot.get_lfd_status and not self._start_record:
                    
                print "started recording"

                self._start_record = True

                # self._record_sample.start_record(self._robot._limb)

                dummy_task = {}

                self._record_sample.start_record(task_action=dummy_task)
                
            elif not self._robot.get_lfd_status and self._start_record:

                print "stopped recording"

                self._start_record = False

                # self._record_sample.stop_record(True)
                self._record_sample.stop_record(task_status=True)
                
                self._finish_demo  = True
                
                option = raw_input('Do you want to record another demo? (y/n)')
                
                #TODO fix this, how to store additional demo without re-starting?
                #here there is a logical error.
                if option == 'y':
                    self._finish_demo  = False
                    self._demo_idx     += 1
                else:
                    self.save_now()

            else:

                #self._robot.get_cuff_state is True or None, don't do anything.

                pass

    def save_now(self):
        self._record_sample.save_data_now()


def main(robot_interface, demo_idx):

    lfd = StoreDemonstration(robot_interface=robot_interface, demo_idx=0)
    
    lfd.save_demo_data()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collect demonstrations')
    
    parser.add_argument('-n', '--demo_index', type=int, help='start index of demo collection')

    parser.add_argument('-l', '--limb_name', type=str, help='limb index-(left/right)')
    
    args = parser.parse_args()

    if args.demo_index is None:

        demo_index = 0

    if args.limb_name is None:

        print "Give limb option, -l left or -l right"
        
        raise ValueError

    rospy.init_node('store_demo_node')

    from aml_robot.baxter_robot import BaxterArm

    arm = BaxterArm(args.limb_name)

    main(robot_interface=arm, demo_idx=0)
