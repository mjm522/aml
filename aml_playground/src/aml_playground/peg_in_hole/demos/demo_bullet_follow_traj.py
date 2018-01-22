import os
import numpy as np
import pybullet as pb
from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config


env = PIHWorld(pih_world_config)

def get_demo():
    """
    load the demo trajectory from the file
    """
    path_to_demo = pih_world_config['demo_folder_path'] + 'pih_js_ee_pos_data_1.csv'

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    demo_data  = np.genfromtxt(path_to_demo, delimiter=',')

    return demo_data


def view_traj(trajectory=get_demo()):
    """
    this funciton is to load trajectory into the bullet viewer.
    the state is a list of 6 values, only the x,y,z values are taken
    """
    for k in range(len(trajectory)-1):

        env.draw_trajectory(point_1=trajectory[k, 6:9], point_2=trajectory[k+1, 6:9], colour=[0,0,1], line_width=2.5)



def apply_control(set_point_pos, set_point_vel, set_point_acc):
    """
    using the inbuilt control commands to make it follow a trajectory
    """

    torque = pb.calculateInverseDynamics(env._robot_id, set_point_pos, set_point_vel, set_point_acc)
    print "Torque computed \t", torque
    
    # pb.setJointMotorControlArray(env._robot_id, 
    #                              env._manipulator._joint_idx, 
    #                              controlMode=pb.TORQUE_CONTROL, 
    #                              forces=torque)

    #gains are terrible
    # pb.setJointMotorControlArray(env._robot_id, 
    #                              env._manipulator._joint_idx, 
    #                              controlMode=pb.VELOCITY_CONTROL,
    #                              targetPositions=set_point_pos,
    #                              targetVelocities=set_point_vel,
    #                              velocityGains=[1.07,1.07,1.07])


    pb.setJointMotorControlArray(env._robot_id, 
                                 env._manipulator._joint_idx, 
                                 controlMode=pb.POSITION_CONTROL,
                                 targetPositions=set_point_pos,
                                 targetVelocities=set_point_vel)



def main():

    view_traj()

    traj2follow = get_demo()
    steps = traj2follow.shape[0]


    for k in range(steps-1):

        set_point_pos = traj2follow[k, :3]
        set_point_vel = traj2follow[k, 3:6]
        set_point_acc = (traj2follow[k+1, 3:6] - traj2follow[k, 3:6])

        print "Trajectory Step \t", k

        error = np.linalg.norm(set_point_pos)

        while error > 0.05:

            env.step()
            
            jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = env._manipulator.get_jnt_state()

            error = np.linalg.norm(jnt_pos-set_point_pos)

            apply_control(set_point_pos.tolist(), set_point_vel.tolist(), set_point_acc.tolist())

            print "Error \t", error

            
 
if __name__ == '__main__':
    main()
