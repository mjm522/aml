import rospy
import numpy as np
import matplotlib.pyplot as plt

from aml_lfd.utilities.utilities import js_inverse_dynamics, get_js_traj, get_sampling_rate, get_effort_sequence_of_demo


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

    torques    = js_inverse_dynamics(limb_name=robot_interface._limb, demo_idx=demo_idx, h_component=False)

    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj(limb_name=robot_interface._limb, demo_idx=demo_idx)
    
    effort_sequence = get_effort_sequence_of_demo(limb_name=robot_interface._limb, demo_idx=demo_idx)

    sampling_rate = get_sampling_rate(limb_name=robot_interface._limb, demo_idx=demo_idx)

    # plot_id(torques)

    rate = rospy.Rate(sampling_rate)

    arm.untuck_arm()

    print "Checking computed position commands ..."
    raw_input('Press any key to continue ...')

    for js_pos in js_pos_traj:

        robot_interface.exec_position_cmd(js_pos)

        rate.sleep()

    raw_input('Completed,  proceeding to torque control, press any key...')

    arm.untuck_arm()

    print "Checking computed torque commands ..."
    raw_input('Press any key to continue ...')

    for tau in torques:

        robot_interface.exec_torque_cmd(tau)

        rate.sleep()

    #to reload the position controllers, else the arms will drift
    robot_interface.exec_position_cmd_delta(np.zeros(7))

    raw_input('Completed,  proceeding to effort control, press any key...')

    arm.untuck_arm()

    print "Checking computed effort commands ..."
    raw_input('Press any key to continue ...')

    for tau in effort_sequence:

        robot_interface.exec_torque_cmd(-tau)

        rate.sleep()

    #to reload the position controllers, else the arms will drift
    robot_interface.exec_position_cmd_delta(np.zeros(7))

if __name__ == '__main__':

    rospy.init_node('id_tester')
    
    from aml_robot.baxter_robot import BaxterArm
    
    limb = 'left'
    
    arm = BaxterArm(limb)

    demo_idx = 0

    main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx)