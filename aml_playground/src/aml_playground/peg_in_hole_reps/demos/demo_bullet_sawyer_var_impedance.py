import numpy as np
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_var_imp_reps import SawyerVarImpREPS
#get the experiment params
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import exp_params



class DummyPolicy():
    """
    this is a dirty hack policy
    when a context is called it
    simply returns from a saved list of 
    values
    """

    def __init__(self, w_list):

        self._count = 0
        self._w_list = w_list

    def compute_w(self, context):
        w = self._w_list[self._count, :]
        self._count += 1
        return w


def test_params():

    data = load_data(exp_params['param_file_name'])
    policy = DummyPolicy(data['w_list'])
    
    # env = SawyerEnv(exp_params['env_params'])
    # env.execute_policy(policy=policy)

    plot_params(data)

    raw_input("Press enter to exit")


def plot_params(data):

    w_list = data['w_list']
    spring_force = data['spring_force']
    force_traj = data['force_traj_local']
    req_traj = data['req_traj']
    pos_traj = data['ee_traj']
    vel_traj = data['ee_vel_traj']
    plt.figure("Varying stiffness plot")
    
    # plt.subplot(3,5,1)
    # plt.title("Kx")
    # plt.plot(w_list[:,0])
    # plt.subplot(3,5,2)
    # plt.title("Px")
    # plt.plot(pos_traj[:,0], 'g')
    # plt.plot(req_traj[:,0], 'r')
    # plt.subplot(3,5,3)
    # plt.title("Vx")
    # plt.plot(vel_traj[:,0])
    # plt.subplot(3,5,4)
    # plt.title("Fx")
    # plt.plot(force_traj[:,0])
    # plt.subplot(3,5,5)
    # plt.title("Sx")
    # plt.plot(spring_force[:,0])

    
    # plt.subplot(3,5,6)
    # plt.title("Ky")
    # plt.plot(w_list[:,1])
    # plt.subplot(3,5,7)
    # plt.title("Py")
    # plt.plot(pos_traj[:,1], 'g')
    # plt.plot(req_traj[:,1], 'r')
    # plt.subplot(3,5,8)
    # plt.title("Vy")
    # plt.plot(vel_traj[:,1])
    # plt.subplot(3,5,9)
    # plt.title("Fy")
    # plt.plot(force_traj[:,1])
    # plt.subplot(3,5,10)
    # plt.title("Sy")
    # plt.plot(spring_force[:,1])

    
    # plt.subplot(3,5,11)
    # plt.title("Kz")
    # plt.plot(w_list[:,2])
    # plt.subplot(3,5,12)
    # plt.title("Pz")
    # plt.plot(pos_traj[:,2], 'g')
    # plt.plot(req_traj[:,2], 'r')
    # plt.subplot(3,5,13)
    # plt.title("Vz")
    # plt.plot(vel_traj[:,2])
    # plt.subplot(3,5,14)
    # plt.title("Fz")
    # plt.plot(force_traj[:,2])
    # plt.subplot(3,5,15)
    # plt.title("Sz")
    # plt.plot(spring_force[:,2])
    # plt.show()

    plt.plot(force_traj[:,2], w_list[:,2])
    plt.show()


def reps():

    rewards = []
    params  = []
    force_penalties = []
    goal_penalties = []

    ps = SawyerVarImpREPS(exp_params)

    plt.figure("Reward plots", figsize=(15,15))
    plt.ion()

    i = 0
    
    while i < exp_params['max_itr']:

        try:

            print "\n\tEpisode \t", i

            policy = ps._gpreps.run()

            s_list, w_list, _, _, reward, traj_data = ps._eval_env.execute_policy(policy=policy,show_demo=False)

            mean_reward = ps._eval_env._penalty['total']
            force_penalty = ps._eval_env._penalty['force']

            print "Parameter found*****************************************: \t", np.mean(w_list, 0)
            print "mean_reward \t", mean_reward

            if mean_reward > -1000:
                rewards.append(mean_reward)

            force_penalties.append(force_penalty)

            goal_penalties.append(force_penalty+mean_reward)
            
            params.append(np.hstack([mean_reward]))

            ps._eval_env._reset()
            plt.clf()
            plt.subplot(311)
            plt.title('Total reward')
            plt.plot(rewards, 'r')
            plt.ylabel("mag")
            plt.subplot(312)
            plt.title('Force felt')
            plt.plot(force_penalties, 'g')
            plt.ylabel("mag")
            plt.subplot(313)
            plt.title('Goal Closeness')
            plt.plot(goal_penalties, 'b')
            plt.xlabel("iterations")
            plt.ylabel("mag")
            plt.pause(0.00001)
            plt.draw()

            i+=1

        except KeyboardInterrupt:
            print "\n\n\t Stopping reps training...\n"
            break

    
    data = {
    'req_traj':traj_data['traj'],
    'spring_force':traj_data['spring_force'],
    'w_list':np.asarray(w_list),
    's_list':np.asarray(s_list),
    'force_traj':traj_data['ee_wrenches'][:,:3],
    'force_traj_local':traj_data['ee_wrenches_local'][:,:3],
    'ee_traj':traj_data['ee_traj'],
    'ee_vel_traj':traj_data['ee_vel_traj'],
    'exp_params':exp_params,
    }

    save_data(data, exp_params['param_file_name'])
    raw_input("Press any key to exit")


def main():

    # reps()
    test_params()


if __name__ == '__main__':
    main()