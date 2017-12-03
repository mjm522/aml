import rospy
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from mpl_toolkits.mplot3d import Axes3D

from aml_io.io_tools import save_data, load_data
from aml_opt.mppi_controller.mppi_controller import MPPIController

from aml_planners.push_planner.dynamics.box2d_dynamics import Box2DDynamics
from aml_planners.push_planner.dynamics.learnt_dynamics import LearnedDynamics

from aml_planners.push_planner.costs.cost_2push_imp import CostImp

from aml_planners.push_planner.utilities.utils import get_circle, get_heatmap, sigmoid
from aml_planners.push_planner.exp_params.experiment_params_2push import experiment_config as econfig

cost = None
dynamics = None
box2d_dynamics = None

try:
    import rospy
except Exception as e:
    rospy_exist = False
else:
    rospy_exist = True

if rospy_exist:
    from geometry_msgs.msg import Point
    from rospy_tutorials.msg import Floats
    from aml_robot.msg import Box2dRosViewerAction
    #write a single message for all these!!!
    publish_box2d_pose   = rospy.Publisher('box2d_world_pose', Point)
    publish_box2d_goal   = rospy.Publisher('box2d_update_goal', Point)
    publish_box2d_action = rospy.Publisher('box2d_action', Box2dRosViewerAction)
    publish_box2d_obstacle = rospy.Publisher('box2d_update_obstacle', Floats)

if econfig['dynamics_type'] == 'learnt_dyn':
    dynamics = LearnedDynamics(econfig)
    box2d_dynamics = Box2DDynamics(econfig)
else:
    dynamics = Box2DDynamics(econfig)
    box2d_dynamics = dynamics


## Cost selection
if econfig['cost_type'] == 'cost_2push_imp':

    cost = CostImp(econfig['mppi_params'])


def get_pose_message(data):
    msg = Point()
    msg.x = data[0]
    msg.y = data[1]
    msg.z = data[2]
    return msg

def get_obstacle_message(data):
    msg = Floats()
    msg.data = data
    return msg

def get_action_message(num_fins, single_action_len, actions):
    msg = Box2dRosViewerAction()
    msg.num_fins=num_fins
    msg.single_action_len=single_action_len
    data = [] 
    for action in actions:
        data += action
    msg.actions = data
    return msg

def transform(x0, angle):
    if isinstance(x0, tuple):
        x0 = np.asarray(x0)
    rot_matrix = np.array([[np.cos(angle),  -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    tmp = np.dot(rot_matrix, np.asarray([x0[0], x0[1]]))
    x0[0] = tmp[0]
    x0[1] = tmp[1]
    return x0


def mppi_run(mppi_params):

    mppi_params['start'] = mppi_params['start']
    mppi_params['goal'] = mppi_params['goal']

    if rospy_exist:
        ####### to send goal to box2d viewer
        msg =  get_pose_message(mppi_params['goal'])
        publish_box2d_goal.publish(msg)

        if mppi_params['obstacle'] is not None:
            msg =  get_obstacle_message(mppi_params['obstacle'])
            publish_box2d_obstacle.publish(msg)
        else:
            publish_box2d_obstacle.publish(Floats())
        #######

    start    = mppi_params['start']
    goal     = mppi_params['goal']

    max_iterations = mppi_params['max_iterations']

    if mppi_params['init_policy'] == 'use_random':
        init_policy = np.asarray([np.random.randn(mppi_params['cmd_dim']) for _ in range(mppi_params['N'])]).squeeze().T
    else:
        init_policy = np.asarray([np.zeros(mppi_params['cmd_dim']) for _ in range(mppi_params['N'])]).squeeze().T

    if mppi_params['cmd_dim'] == 1:
        init_policy = init_policy[None, :]

    is_quasi_static = econfig['is_quasi_static']

    task_complete = False

    mppi_controller = MPPIController(dynamics, cost, mppi_params)
    
    #####################TO match the data#############################################
    x0 = start.copy()

    us = init_policy.copy()

    trajectory = [x0]
    dist_to_goal = [np.linalg.norm(x0-goal)]

    plt.ion()

    itr = 0
    total_dist_to_goal = np.linalg.norm(mppi_controller._x0-goal)
    
    while not task_complete:                

        if rospy_exist:
            task_complete =  task_complete or rospy.is_shutdown()
            
            ####### to send to box2d viewer
            msg =  get_pose_message(x0)
            publish_box2d_pose.publish(msg)
            #######

        mppi_controller.init(x0)

        plt.figure("Box position")
        plt.clf()

        us, xs_samples, ss, delus_samples = mppi_controller.per_step_iteration(u_list=us)

        action = us[:,0]

        print "The action I am going to take is :=\t", sigmoid(action)

        if is_quasi_static:
            x0, _, converted_action = box2d_dynamics(np.array([x0[0],x0[1],0,0,0,0]), action)
        else:
            x0, _, converted_action = box2d_dynamics(x0, action)

        if rospy_exist:
            ####### to send to box2d viewer
            msg =  get_action_message(num_fins=2, single_action_len=7, actions=converted_action)
            publish_box2d_action.publish(msg)
            #######

        us = np.roll(us, -1, axis=1) #shift the control sequence
        us[:,-1] = np.random.randn(mppi_params['cmd_dim'])

        # Plotting stuff
        tmp = np.asarray(trajectory)
        plt.scatter(tmp[:,0], tmp[:,1])

        plt.scatter(start[0],start[1],c='r', s=100, marker="*")
        plt.scatter(goal[0],goal[1],  c='g', s=110, marker="*")

        dist = np.linalg.norm(x0[:2]-goal[:2])

        dist_to_goal.append(dist)

        print "Distance to goal \t", dist

        if len(trajectory) >= 1:

            plt.figure("Distance to goal")
            plt.clf()
            plt.plot(dist_to_goal)

            plt.draw()
            plt.pause(0.00001)

            # #if within a small bound of the final state, then the task is complete
            if dist < mppi_params['task_complete_thresh']:
                task_complete = True
                print "Total iterations for MPPI controller is \t", mppi_controller._itr

            curr_dist_to_goal = np.linalg.norm(mppi_controller._x0-goal)

            itr += 1

            if itr > max_iterations:
                task_complete = True
                print "What is that man?"


def main():

    if rospy_exist:
        rospy.init_node("baxter_experiment_node")
    
    mppi_params = econfig['mppi_params']

    mppi_run(mppi_params)

    raw_input("waiting...")

if __name__ == '__main__':
    main()