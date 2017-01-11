import rospy
import numpy as np
import matplotlib.pyplot as plt

from aml_lfd.utilities.utilities import js_inverse_dynamics


def plot_id(torques):
    plt.figure()
    plt.title('Joint space trajectories')

    num_jnts = torques.shape[1]
    plot_no = num_jnts*100 + 11 #just for getting number of joints

    for i in range(num_jnts):
        plt.subplot(plot_no)
        plt.plot(torques[:,i])
        plot_no += 1

    plt.show()

def main(robot_interface, load_from_demo, demo_idx):

    # torques    = js_inverse_dynamics(limb_name=limb, demo_idx=demo_idx, h_component=True)
    torques    = js_inverse_dynamics(limb_name=limb, demo_idx=demo_idx, h_component=False)

    rate = rospy.Rate(50)

    for tau in torques:

        robot_interface.exec_torque_cmd(tau)

        rate.sleep()

if __name__ == '__main__':

    rospy.init_node('id_tester')
    
    from aml_robot.baxter_robot import BaxterArm
    
    limb = 'left'
    
    arm = BaxterArm(limb)

    # arm.untuck_arm()

    demo_idx = 0

    main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx)