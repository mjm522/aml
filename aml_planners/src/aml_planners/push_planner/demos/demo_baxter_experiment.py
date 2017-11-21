import scipy.misc
import numpy as np
import sys, os, copy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point

from aml_opt.mppi_controller.mppi_controller import MPPIController
from aml_planners.push_planner.dynamics.box2d_dynamics import Box2DDynamics
from aml_planners.push_planner.dynamics.learnt_dynamics import LearnedDynamics
from aml_planners.push_planner.push_worlds.baxter_push_world import BaxterPushWorld
from aml_planners.push_planner.exp_params.experiment_params import experiment_config as econfig


from aml_planners.push_planner.costs.cost_imp import CostImp
from aml_planners.push_planner.utilities.utils import sigmoid

from aml_perception.camera_sensor import CameraSensor

import tf, rospy
plt.ion()

camera_sensor = CameraSensor()

# world = BaxterPushWorld(transform=False)
# while not rospy.is_shutdown():
#     print "World start \n", world.get_state()
# print shds

if econfig['dynamics_type'] == 'learnt_dyn':
    dynamics = LearnedDynamics(econfig, add_extra_dim=True)
    world = BaxterPushWorld(transform=False)
else:
    dynamics = Box2DDynamics(econfig)
    world = BaxterPushWorld(transform=True)

## Cost selection
if econfig['cost_type'] == 'cost_imp':

    cost = CostImp(econfig['mppi_params'])
else:
    raise("Unknown cost type")


def get_pose_message(data):
    msg = Point()
    msg.x = data[0]
    msg.y = data[1]
    msg.z = data[2]
    return msg



def mppi_run_ros(mppi_params):

    image_index = 0

    rospy.init_node("baxter_experiment_node")

    publish_box2d_pose = rospy.Publisher('box2d_world_pose', Point)

    task_complete = False

    if econfig['dynamics_type'] == 'learnt_dyn':
        transform_angle = 0.
    else:
        transform_angle = -0.5*np.pi

    world_start = np.r_[world.get_state(),0.,0.,0]

    #### send to the box2d
    msg =  get_pose_message(world_start.copy())
    publish_box2d_pose.publish(msg)
    ####
    
    mppi_params['start'] = world_start.copy()

    if world._transform:
        mppi_params['goal']  = world.baxter_2_box_2d_transform(mppi_params['goal'])

    mppi_controller = MPPIController(dynamics, cost, mppi_params)

    init_policy = np.ones([mppi_params['cmd_dim'], mppi_params['N']]) * 0.675
    # init_policy = np.random.rand(mppi_params['cmd_dim'], mppi_params['N'])*0.

    x0 = mppi_params['start'].copy()

    us = init_policy.copy()

    trajectory = [x0]
    dist_to_goal = [np.linalg.norm(x0-mppi_params['goal'])]

    if us.ndim == 1:
        us = us[None,:]

    ignore_first_push = True

    while not task_complete:

        mppi_controller.init(x0)

        ## Given current state x0, optimise over future horizon

        # us, xs_samples, ss, delus_samples = mppi_controller.fun_to_parrallelize(u_list=us)

        us, xs_samples, ss, delus_samples = mppi_controller.per_step_iteration(u_list=us)

        ## Action to be executed is the first one
        action = np.copy(sigmoid(us[:,0]))#np.multiply(econfig['cmd_filter'],us[:,0])

        print "The action I am going to take is :=\t", action

        print "State: \n", np.round(world.get_state(),3)
        print "Goal:\n", np.round(mppi_params['goal'],3)

        # raw_input("Procced?  Press enter to continue")
    
        # # print "Action: ", action
        
        if ignore_first_push:
            ignore_first_push = False
        else:
            sucess = world.apply_push(action)
            # image_index += 1
            # save_image(image_index)

        # raw_input("Enter to continue...")
        x0 = np.r_[world.get_state(),0.,0.,0]

        ####### to send to box2d viewer
        msg =  get_pose_message(x0)
        publish_box2d_pose.publish(msg)
        #######

        if mppi_params['goal_cost_coeff'] == 0.:
            dist = np.linalg.norm(x0[2]-mppi_params['goal'][2])
        else:
            dist = np.linalg.norm(x0[:2]-mppi_params['goal'][:2])

        if dist < mppi_params['task_complete_thresh']:
            task_complete = True

        dist_to_goal.append(dist)
        print "Distance to the goal \t", dist

        plt.figure("Distance to goal")
        plt.clf()
        plt.plot(dist_to_goal)
        plt.draw()
        plt.pause(0.00001)

        # sigma = mppi_controller.get_unc(x0, np.zeros(mppi_params['cmd_dim']), mppi_controller._dt, 5, 0.25)
        # sigma = np.sum(np.sqrt(sigma))/6.0

        # x0[3:] = 0. #quasi static behaviour
        # trajectory.append(x0)

        # us = init_policy.copy()

        us = np.roll(us, -1, axis=1) #shifting left the control sequence
        us[:,-1] = np.random.randn(mppi_params['cmd_dim']) #np.zeros(mppi_params['cmd_dim'])#

    if task_complete:

        for _ in range(3):
            os.system("spd-say 'Dear Human, Hurray... I reached the goal...'")


def main():

    mppi_params = econfig['mppi_params']

    mppi_run_ros(mppi_params)
        


if __name__ == '__main__':
    main()