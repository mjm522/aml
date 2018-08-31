import os
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rl_algos.agents.creps_new import CREPSOpt
from aml_io.io_tools import save_data, load_data
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
# from rl_algos.policy.neural_network_policy import NeuralNetPolicy
from aml_rl_envs.point_mass.point_mass_env import PointMassEnv
from aml_playground.utilities.generate_document import create_experiment_document
from aml_playground.var_imp_reps.policy.spring_init_policy import create_init_policy
from aml_playground.var_imp_reps.exp_params.experiment_point_mass_params import exp_params

np.random.seed(123)

root_save_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/point_mass/'

kd_scales = range(2,7,1)
timesteps = [x/100.0 for x in range(5,51,5)]
force_model_enabled = [True, False]


def exp_plot(rewards, reward_traj, traj_draw, save_path=None, plot_live=True, figsize=(10,10)):

    params = np.asarray(traj_draw['params'])

    u_list = np.asarray(traj_draw['u_list'])

    ee_traj = np.asarray(traj_draw['ee_traj'])

    req_traj = np.asarray(traj_draw['traj'])

    ee_vel_traj = np.asarray(traj_draw['ee_vel_traj'])

    ####################PLOT 1############################
    plt.figure('Reward', figsize=figsize)
    plt.plot(rewards, 'b')


    ####################PLOT 2############################
    plt.figure('Kp-Kd-u', figsize=figsize)
    if exp_params['env_params']['z_only']:
        plt.subplot(3,3,1)
        plt.cla()
        plt.title('Kp x')

        plt.subplot(3,3,2)
        plt.cla()
        plt.title('Kp y')

        plt.subplot(3,3,3)
        plt.cla()
        plt.title('Kp z')
        plt.plot(params[:,0])

        plt.subplot(3,3,4)
        plt.cla()
        plt.title('Kd x')

        plt.subplot(3,3,5)
        plt.cla()
        plt.title('Kd y')

        plt.subplot(3,3,6)
        plt.cla()
        plt.title('Kd z')
        plt.plot(params[:,1])

    else:
        plt.subplot(3,3,1)
        plt.cla()
        plt.title('Kp x')
        plt.plot(params[:,0])

        plt.subplot(3,3,2)
        plt.cla()
        plt.title('Kp y')
        plt.plot(params[:,1])

        plt.subplot(3,3,3)
        plt.cla()
        plt.title('Kp z')
        plt.plot(params[:,2])

        plt.subplot(3,3,4)
        plt.cla()
        plt.title('Kd x')
        plt.plot(params[:,3])

        plt.subplot(3,3,5)
        plt.cla()
        plt.title('Kd y')
        plt.plot(params[:,4])

        plt.subplot(3,3,6)
        plt.cla()
        plt.title('Kd z')
        plt.plot(params[:,5])

    plt.subplot(3,3,7)
    plt.cla()
    plt.title('U x')
    plt.plot(u_list[:,0])

    plt.subplot(3,3,8)
    plt.cla()
    plt.title('U y')
    plt.plot(u_list[:,1])

    plt.subplot(3,3,9)
    plt.cla()
    plt.title('U z')
    plt.plot(u_list[:,2])

    ####################PLOT 3############################
    plt.figure('Traj-Vel', figsize=figsize)
    plt.subplot(2,3,1)
    plt.cla()
    plt.plot(req_traj[:,0], 'r')
    plt.plot(ee_traj[:,0], 'g')
    plt.title('traj x')

    plt.subplot(2,3,2)
    plt.cla()
    plt.plot(req_traj[:,1], 'r')
    plt.plot(ee_traj[:,1], 'g')
    plt.title('traj y')

    plt.subplot(2,3,3)
    plt.cla()
    plt.plot(req_traj[:,2], 'r')
    plt.plot(ee_traj[:,2], 'g')
    plt.title('traj z')

    plt.subplot(2,3,4)
    plt.cla()
    plt.plot(ee_vel_traj[:,0])
    plt.title('vel x')

    plt.subplot(2,3,5)
    plt.cla()
    plt.plot(ee_vel_traj[:,1])
    plt.title('vel y')

    plt.subplot(2,3,6)
    plt.cla()
    plt.plot(ee_vel_traj[:,2])
    plt.title('vel z')

    ####################PLOT 4############################
    plt.figure('Reward traj', figsize=figsize)
    plt.subplot(6,1,1)
    plt.cla()
    plt.plot(reward_traj['u_reward_traj'], 'b')
    plt.title('u_reward_traj')
    plt.subplot(6,1,2)
    plt.cla()
    plt.plot(reward_traj['delta_u_reward_traj'], 'b')
    plt.title('delta_u_reward_traj')
    plt.subplot(6,1,3)
    plt.cla()
    plt.plot(reward_traj['param_reward_traj'], 'b')
    plt.title('param_reward_traj')
    plt.subplot(6,1,4)
    plt.cla()
    plt.plot(reward_traj['delta_param_reward_traj'], 'b')
    plt.title('delta_param_reward_traj')
    plt.subplot(6,1,5)
    plt.cla()
    plt.plot(reward_traj['goal_reward_traj'], 'b')
    plt.title('goal reward')
    plt.subplot(6,1,6)
    plt.cla()
    plt.plot(reward_traj['reward_traj'], 'b')
    plt.title('total')

    ####################PLOT 5############################
    plt.figure("Penalty traj", figsize=figsize)
    plt.subplot(4,1,1)
    plt.cla()
    plt.plot(reward_traj['goal_pos_penalty'], 'b')
    plt.title('goal_pos_penalty')
    plt.subplot(4,1,2)
    plt.cla()
    plt.plot(reward_traj['goal_delta_pos_penalty'], 'b')
    plt.title('goal_delta_pos_penalty')
    plt.subplot(4,1,3)
    plt.cla()
    plt.plot(reward_traj['goal_vel_penalty'], 'b')
    plt.title('goal_vel_penalty')
    plt.subplot(4,1,4)
    plt.cla()
    plt.plot(reward_traj['goal_penalty'], 'b')
    plt.title('goal_penalty')

    if plot_live:
        plt.draw()
        plt.pause(0.0001)

    if save_path is not None:

        plt.figure('Reward').savefig(save_path+'/Reward')
        plt.figure('Kp-Kd-u').savefig(save_path+'/Kp-Kd-u')
        plt.figure('Traj-Vel').savefig(save_path+'/Traj-Vel')
        plt.figure('Reward traj').savefig(save_path+'/Reward traj')
        plt.figure('Penalty traj').savefig(save_path+'/Penalty traj')


def search_hyper_params(plot_live = False):

    for scale_val in kd_scales:

        for timestep_val in timesteps:

            for bool_val in force_model_enabled:

                kd_scale = scale_val
                kp_scale = (kd_scale**2.)/4.

                exp_params_local = copy.deepcopy(exp_params)

                exp_params_local['env_params']['param_scale'] = np.array([ kp_scale, kp_scale, kp_scale, kd_scale,  kd_scale,  kd_scale ])
                exp_params_local['env_params']['time_step'] = timestep_val

                if not bool_val:
                    exp_params_local['env_params']['force_predict_model'] = None

                print "\n\n\n**********************************************************************"
                print "\tkd_scale = %d, timestep = %f, force_model_enabled = %s\n\n**********************************************************************\n\n"%(kd_scale, timestep_val, bool_val)

                experiment_name = "kd_scale_%d_tsteps_%f_ff_%s"%(kd_scale, timestep_val, bool_val)

                save_path = root_save_path + experiment_name

                if not os.path.exists(save_path):
                    os.makedirs(save_path)                

                rewards, reward_traj, traj_draw = do_the_experiment(exp_params_local, plot_live)

                exp_plot(rewards=rewards, reward_traj=reward_traj, traj_draw=traj_draw, save_path=save_path, plot_live=plot_live)

                plt.close('all')

                create_experiment_document(exp_name=experiment_name, image_folder=save_path+'/', exp_params=exp_params_local)

                for k in range(time_steps):
                    w_list[:,:,k] = policy[k]._w
                    sigma_list[:,:,k] = policy[k]._sigma

                data[-1]['last_w_list'] = w_list
                data[-1]['last_sigma_list'] = sigma_list

                c='y'#raw_input("Save data? (y/N)")

                if c == 'y':
                    print "Saving to %s"%exp_params_local['param_file_name']
                    save_data(data, save_path+'/data_file.pkl')


def do_the_experiment(exp_params_local, plot_live=True):

    random_state = np.random.RandomState(0)
    initial_params = .001 * np.ones(exp_params_local['gpreps_params']['w_dim'])
    num_policy_updates = 20
    n_samples_per_update = 20
    variance = 0.03
    time_steps=exp_params_local['time_steps']

    policy_per_time_step = exp_params_local['gpreps_params']['policy_per_time_step']

    rewards = []

    env_params = exp_params_local['env_params']
    env_params['renders'] = False

    env = PointMassEnv(env_params)

    env_params['renders'] = False
    trail_env = PointMassEnv(env_params)

    # sess=tf.Session()

    if policy_per_time_step:
        policy = [ LinGaussPolicy(w_dim=exp_params_local['gpreps_params']['w_dim'], context_feature_dim=exp_params_local['gpreps_params']['context_feature_dim'], variance=0.03, covariance_scale=1.0, initial_params=initial_params, bounds=exp_params_local['gpreps_params']['w_bounds'], random_state=random_state, transform=False) for _ in range(time_steps)]
    else: 
        policy = LinGaussPolicy(w_dim=exp_params_local['gpreps_params']['w_dim'], context_feature_dim=exp_params_local['gpreps_params']['context_feature_dim'], variance=0.03, covariance_scale=1.0, initial_params=initial_params, bounds=exp_params_local['gpreps_params']['w_bounds'], random_state=random_state, transform=False)

    # policy = [ NeuralNetPolicy(sess=sess, w_dim=exp_params['gpreps_params']['w_dim'], context_feature_dim=exp_params['gpreps_params']['context_feature_dim'], bounds=exp_params['gpreps_params']['w_bounds'], learning_rate=0.01, scope="policy_estimator_%s"%i) for i in range(time_steps)]

    # kp_traj = np.zeros_like(env._des_force_traj)
    # kp_traj[:,2] = env._des_force_traj[:,2]

    # kd_traj = np.ones_like(env._des_force_traj)*2
    # kd_traj[:,2] = np.sqrt(env._des_force_traj[:,2])

    # ctrl_traj = np.hstack([kp_traj,kd_traj])

    # policy = create_init_policy(env._traj2pull, env._des_force_traj, ctrl_traj, exp_params, set_frm_data=exp_params['start_policy'])

    if exp_params['env_params']['z_only']:
        w_list = np.zeros([2,3,time_steps])
        sigma_list = np.zeros([2,2,time_steps])
    else:
        w_list = np.zeros([6,9,time_steps])
        sigma_list = np.zeros([6,6,time_steps])

    mycreps = CREPSOpt(entropy_bound=exp_params_local['gpreps_params']['entropy_bound'], initial_params=initial_params, num_policy_updates=num_policy_updates, num_samples_per_update=n_samples_per_update, num_old_datasets=1, env=env, policy=policy, transform_context=False, policy_per_time_step = policy_per_time_step, time_steps=time_steps)

    if plot_live:
        plt.ion()

    data = []

    it = 0
    while it < (exp_params_local['n_episodes']):

        try:

            print "Episode \t", it

            policy = mycreps.run(smooth_policy=exp_params_local['smooth_policy'], jnt_space=False)

            trail_env._reset()

            traj_draw, reward_traj = trail_env.execute_policy(policy=policy, explore=False, jnt_space=False, policy_per_time_step = policy_per_time_step)
            
            rewards.append(reward_traj['total'])

            traj_draw['mean_reward'] = reward_traj['total']

            data.append(traj_draw)

            it += 1

            if plot_live:
                exp_plot(rewards=rewards, reward_traj=reward_traj, traj_draw=traj_draw, plot_live=plot_live, figsize=(5,5))

        except KeyboardInterrupt:
            break

    return rewards, reward_traj, traj_draw



def main():
    do_the_experiment(exp_params_local=exp_params)



if __name__ == '__main__':
    main()

raw_input("press any key to continue")