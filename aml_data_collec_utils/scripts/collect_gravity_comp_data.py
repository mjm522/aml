import numpy as np
import random
import rospy

class CollectGravityCompData():
    
    def __init__(self, robot_interface, data_cnt=10, sample_rate=50):
        self._robot =  robot_interface
        self._jnt_limits = self._robot._jnt_limits
        self._data_cnt = data_cnt
        self._input_data = []
        self._output_data = []
        self._robot.set_sampling_rate(sampling_rate=sample_rate)
        #this enables to collect more data within data_cnt interwal, specifically data_cnt*sample_rate
        update_period = rospy.Duration(1.0/sample_rate)
        rospy.Timer(update_period, self.save_gravity_comp_data)

    def choose_random_jnt_confgn(self):
        rand_jnt_pos = np.zeros(self._robot._nu)
        for jnt in range(self._robot._nu):
            rand_jnt_pos[jnt] = random.uniform(self._jnt_limits[jnt]['lower'], self._jnt_limits[jnt]['upper'])

        return rand_jnt_pos

    def save_gravity_comp_data(self, event):
        self._input_data.append(np.hstack([self._robot._state['position'], self._robot._state['velocity']]))
        self._output_data.append(self._robot._h)

    def collect_data(self):

        for k in range(self._data_cnt):
            cmd = self.choose_random_jnt_confgn()
            self._robot.move_to_joint_pos(cmd)

        np.savetxt('gravity_comp_input_data.txt',  np.asarray(self._input_data).squeeze())
        np.savetxt('gravity_comp_output_data.txt', np.asarray(self._output_data).squeeze())


def main(robot_interface):

    cgcd = CollectGravityCompData(robot_interface=robot_interface)
    cgcd.collect_data()

if __name__ == '__main__':
    rospy.init_node('classical_postn_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    main(arm)
