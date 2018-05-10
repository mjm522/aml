import rospy
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from mpl_toolkits.mplot3d import Axes3D

from aml_io.io_tools import save_data, load_data
# from aml_opt.mppi_controller.mppi_controller import MPPIController
from aml_opt.mppi_controller.mppi_controller_new import MPPIController

from aml_planners.push_planner.dynamics.box2d_dynamics import Box2DDynamics
from aml_planners.push_planner.dynamics.learnt_dynamics import LearnedDynamics

from aml_planners.push_planner.costs.cost_imp import CostImp
from aml_planners.push_planner.costs.cost_scl_unc import CostScaledUnc
from aml_planners.push_planner.costs.cost_traj_following import CostTrajFollowing

from aml_planners.push_planner.utilities.utils import get_circle, get_heatmap, sigmoid
from aml_planners.push_planner.exp_params.experiment_params import experiment_config as econfig

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
if econfig['cost_type'] == 'cost_imp':

    cost = CostImp(econfig['mppi_params'])

elif econfig['cost_type'] == 'cost_traj_following':

    cost = CostTrajFollowing(econfig['mppi_params'])
    cost._curr_goal_index = 25
    goal_pos = cost._traj[2,:]
    econfig['mppi_params']['start'] = np.array([goal_pos[0],goal_pos[1],0,0,0,0])

elif econfig['cost_type'] == 'cost_scl_unc':
    
    cost = CostScaledUnc(econfig['mppi_params'])

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
    # if isinstance(x0, np.ndarray):
    #     x0 = tuple(x0)
    return x0


TYPE = 'EXPERIMENT' #'EXPERIMENT2'

def mppi_run(mppi_params):

    print "Unc Gain is: ", mppi_params['uncertain_cost_coeff']

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

    ## Not experimented yet
    update_rho = mppi_params['update_rho']
    update_RhoN = mppi_params['update_RhoN']
    rhoMin = mppi_params['rhoMin']
    rhoMax = mppi_params['rhoMax']

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

    print "Current rho_value \t", mppi_controller._rho

    total_unc = []
    traj_unc_mean = []
    traj_unc_sigma = []
    traj_cost = []

    evaluation = {
    'traj_unc_mean':None,
    'traj_unc_sigma':None,
    'traj_cost':None,
    'traj': None,
    'exceed_max_itr': None,
    }

    heatmap = None

    if mppi_params['dynamics_type'] == 'learnt_dyn':
        heatmap = get_heatmap(model=dynamics._model, cmd_dim= mppi_params['cmd_dim'], cost=None, obstacle=None,  r_obs=None)
    #     save_data(heatmap, '../../../planner/heatmap.pkl')
    
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

        # us, xs_samples, ss, delus_samples = mppi_controller.fun_to_parrallelize(u_list=us)

        us, xs_samples, ss, delus_samples = mppi_controller.per_step_iteration(u_list=us)

        action = us[:,0]

        print "The action I am going to take is :=\t", sigmoid(action)

        if is_quasi_static:
            x0, _, converted_action = box2d_dynamics(np.array([x0[0],x0[1],0,0,0,0]), action)
        else:
            x0, _, converted_action = box2d_dynamics(x0, action)

        if rospy_exist:
            ####### to send to box2d viewer
            msg =  get_action_message(num_fins=1, single_action_len=7, actions=converted_action)
            publish_box2d_action.publish(msg)
            #######

        sigma = mppi_controller.get_unc(x0, np.zeros(mppi_params['cmd_dim']), mppi_controller._dt, 5, 0.25)
        # sigma = np.sum(np.sqrt(sigma))/6.0

        total_unc.append(sigma)

        trajectory.append(x0)

        us = np.roll(us, -1, axis=1) #shift the control sequence
        us[:,-1] = np.random.randn(mppi_params['cmd_dim'])

        # Plotting stuff

        if heatmap is not None:
            plt.imshow(heatmap, origin='lower', interpolation='none', extent=[0,1.,0,1.])
            plt.colorbar()
        
        tmp = np.asarray(trajectory)
        plt.scatter(tmp[:,0], tmp[:,1])

        plt.scatter(start[0],start[1],c='r', s=100, marker="*")
        plt.scatter(goal[0],goal[1],  c='g', s=110, marker="*")

        dist = np.linalg.norm(x0[:2]-goal[:2])

        if isinstance(cost,CostTrajFollowing):

            plt.scatter(cost._traj[:,0],cost._traj[:,1],c='m')

            dst = np.linalg.norm(cost.get_goal()-x0[:2])

            if dst <= 1.5:
                cost.next_goal()

        dist_to_goal.append(dist)

        print "Distance to goal \t", dist

        # print "Path Steps: ", len(trajectory)
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

            # if (itr+1)%update_RhoN == 0:
            #     mppi_controller._rho = max(0.1, mppi_controller._rho/2.)

            curr_dist_to_goal = np.linalg.norm(mppi_controller._x0-goal)


            itr += 1

            if itr > max_iterations:
                task_complete = True
                print "What is that man?"

            traj_cost.append(np.sum(np.sum(ss, axis=0)/mppi_params['K'], axis=0))
            #can we change these coversions to a fixed memory allocation based?
            traj_unc_mean.append(np.mean(np.asarray(total_unc)))
            traj_unc_sigma.append(np.sum(np.asarray(total_unc)))

    evaluation = {
    'traj_unc_mean':np.asarray(traj_unc_mean),
    'traj_unc_sigma':np.asarray(traj_unc_sigma),
    'traj_cost':np.asarray(traj_cost),
    'traj': np.asarray(trajectory),
    'exceed_max_itr': itr > max_iterations,
    }

    return evaluation


def main():

    if rospy_exist:
        rospy.init_node("baxter_experiment_node")
    
    global TYPE

    mppi_params = econfig['mppi_params']

    if TYPE == 'TUNING':        
        mppi_params, updated_params = search_mppi_params(params=mppi_params, evaluator=mppi_run, max_iter=5)
        
        #save only if the parameters where updated
        if updated_params:
            file_name = './hyperparams'
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            file_name += '/param_' + strftime("%Y%m%d_%H%M%S", gmtime()) + '.pkl'
            save_data(mppi_params, file_name)

    elif TYPE == 'EXPERIMENT2':
        file_names = ['param_20170620_180523.pkl']

        for file_name in file_names:              
            # mppi_params = load_data('./hyperparams/'+file_name)
            #[0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:#
            #[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0]:
            #0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
            for unc in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105,110,115,120,125,130,135,140,145,150]:#[15.0, 25.0, 35.0, 40.0, 45.0, 50.0, 55.0, 65.0, 70.0, 75.0]:
                mppi_params['uncertain_cost_coeff'] = unc
                global cost
                cost = CostImp(mppi_params)
                evaluation = mppi_run(mppi_params)

                plt.figure(1)
                if econfig['savefig']:
                    plt.savefig("experiment_out/experiment2_%f.png"%(unc,))

                print "The cost of last iteration \t", np.sum(evaluation['traj_cost'])
                print "Evaluation: "
                print "Unc: ", np.sum(evaluation['traj_unc_mean']), np.mean(evaluation['traj_unc_mean'])
    elif TYPE == 'EXPERIMENT':
        evaluation = mppi_run(mppi_params)

        print "The cost of last iteration \t", np.sum(evaluation['traj_cost'])
        print "Evaluation: "
        print "Unc: ", np.sum(evaluation['traj_unc_mean']), np.mean(evaluation['traj_unc_mean'])
        raw_input("waiting...")

if __name__ == '__main__':
    main()