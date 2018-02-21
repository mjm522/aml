#!/usr/bin/env python
import rospy
from config import collect_robot_data_config
from aml_robot.baxter_robot import BaxterArm
from aml_data_collec_utils.core.data_recorder import DataRecorder

class Task(object):
    """
    This is a dummy task interface class.
    this sort of class can be specified to the sample recorder
    incase additional data of an object the robot iteracts needs to be collected
    """
    def __init__(self):
        pass

    def get_status(self):

        return True

    def get_effect(self):

        dummy_task = {}

        return dummy_task


class CollectRobotData(object):
    """
    The class that collects all data regarding a robot
    To use the sample recorder, do the following.
    Turn on the robot, get the lfd button status of the robot (e.g. baxter)
    when the robot arm is moving this value is usually unset.
    When the arm stops it generates an interrupt which is caught by the
    code and the recording stops.
    """

    def __init__(self, robot_interface, config):
        """
        Constructor of the class
        Args:
        robot_interface : the handle to the robot object
        config: params:
                        data_folder_path: folder in which data has to be stored
                        data_name_prefix: specific name of each data sample
                        num_samples_per_file : number of samples per file
                        sampling_rate: at what rate data needs to be sampled from the robot
        """

        self._robot = robot_interface

        self._robot.set_sampling_rate(sampling_rate=config['sampling_rate'])

        self._sample_recorder = DataRecorder(robot_interface=self._robot, 
                                           task_interface=Task(),
                                           data_folder_path=config['data_folder_path'],
                                           data_name_prefix=config['data_name_prefix'],
                                           num_samples_per_file=config['num_samples_per_file'])


        print "Data storage path is:=\t", config['data_folder_path']

        self._start_record = False

        self._finish_demo  = False


    def collect_data(self):

        while (not self._finish_demo) and (not rospy.is_shutdown()):

            print "Robot cuff state \t", self._robot.get_lfd_status

            if self._robot.get_lfd_status and not self._start_record:
                    
                print "started recording"

                self._start_record = True

                dummy_task = {}

                #start recording the data
                self._sample_recorder.start_record(task_action=dummy_task)
                
            elif not self._robot.get_lfd_status and self._start_record:

                print "stopped recording"

                self._start_record = False

                #stop recording the data
                self._sample_recorder.stop_record(task_status=True)
                
                self._finish_demo  = True
                
                option = raw_input('Do you want to record another demo? (y/n)')
                
                #TODO fix this, how to store additional demo without re-starting?
                #here there is a logical error.
                if option == 'y':
                    self._finish_demo  = False
                else:
                    self.save_data_now()

            else:

                #self._robot.get_cuff_state is True or None, don't do anything.

                pass

    def save_data_now(self):

        self._sample_recorder.save_data_now()


    def run(self):

        self.collect_data()



def main():

    rospy.init_node('collect_robot_data', anonymous=True)

    limb = 'left'
    arm = BaxterArm(limb)

    crd = CollectRobotData(robot_interface=arm, config=collect_robot_data_config)

    crd.run()


if __name__ == '__main__':
    main()
