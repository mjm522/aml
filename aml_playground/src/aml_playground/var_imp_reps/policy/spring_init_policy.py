import os
import numpy as np
from aml_io.io_tools import load_data
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy

np.random.seed(123)
random_state = np.random.RandomState(0)

def create_init_policy(given_traj, given_force_traj, given_ctrl_traj, exp_params, set_frm_data=None):

    num_steps = given_traj.shape[0]

    state_dim = 9 ## ----- x, xd, f
    ctrl_dim = 6 ## ----- 3x Kp, 3x Kd

    policy_seq = np.zeros((num_steps, ctrl_dim, state_dim))    

    vel = np.vstack([np.diff(given_traj, axis = 0)/0.01,np.array([0,0,0])])

    state_matrix = np.hstack([given_traj,vel,given_force_traj])

    policy = [ LinGaussPolicy(w_dim=exp_params['gpreps_params']['w_dim'], context_feature_dim=9, variance=0.03, 
                         initial_params=.001 * np.ones(exp_params['gpreps_params']['w_dim']), bounds=exp_params['gpreps_params']['w_bounds'],   random_state=random_state, transform=False) for _ in range(num_steps)]

    if set_frm_data is not None:
        data = load_data(set_frm_data)

    for i in range(num_steps):

        if  set_frm_data is None:

            s = np.reshape(state_matrix[i,:],[state_dim,1])
            f = np.reshape(given_ctrl_traj[i,:], [ctrl_dim,1])

            # print np.dot(s,s.T)

            s_st_inv = np.linalg.pinv(np.dot(s,s.T))

            policy_seq[i,:] = np.dot(f, np.dot(s.T, s_st_inv))

            policy[i]._w = policy_seq[i,:]
            policy[i]._sigma = np.eye(exp_params['gpreps_params']['w_dim'])*0.1

        else:
            policy[i]._w = data[-1]['last_w_list'][:,:,i]
            policy[i]._sigma = data[-1]['last_sigma_list'][:,:,i]

    return policy


if __name__ == '__main__':
    


    from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
    from aml_playground.var_imp_reps.exp_params.experiment_var_imp_params import exp_params

    env_params = exp_params['env_params']
    env_params['renders'] = False

    env = SawyerEnv(env_params)


    # control_traj = np.hstack([env._des_force_traj,])
    kp_traj = np.zeros_like(env._des_force_traj)
    kp_traj[:,2] = env._des_force_traj[:,2]

    kd_traj = np.ones_like(env._des_force_traj)*2
    kd_traj[:,2] = np.sqrt(env._des_force_traj[:,2])

    ctrl_traj = np.hstack([kp_traj,kd_traj])

    policy_seq = create_init_policy(env._traj2pull, env._des_force_traj, ctrl_traj)

    num_steps, _, _ = policy_seq.shape

    check_policy = np.zeros(num_steps)

    vel = np.vstack([np.diff(env._traj2pull, axis = 0)/0.01,np.array([0,0,0])])

    state_matrix = np.hstack([env._traj2pull,vel, env._des_force_traj])

    for i in range(num_steps):
        tmp = np.dot(policy_seq[i,:,:], state_matrix[i,:])

        check_policy[i] = np.linalg.norm(tmp-ctrl_traj[i,:])

    print check_policy

