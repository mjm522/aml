import numpy as np


def create_init_policy(given_traj, given_force_traj, given_ctrl_traj):

    num_steps = given_traj.shape[0]

    state_dim = 9 ## ----- x, xd, f
    ctrl_dim = 6 ## ----- 3x Kp, 3x Kd

    policy_seq = np.zeros((num_steps, ctrl_dim, state_dim))    

    vel = np.vstack([np.diff(given_traj, axis = 0)/0.01,np.array([0,0,0])])

    state_matrix = np.hstack([given_traj,vel,given_force_traj])

    for i in range(num_steps):

        s = np.reshape(state_matrix[i,:],[state_dim,1])
        f = np.reshape(given_ctrl_traj[i,:], [ctrl_dim,1])

        # print np.dot(s,s.T)

        s_st_inv = np.linalg.pinv(np.dot(s,s.T))

        policy_seq[i,:] = np.dot(f, np.dot(s.T, s_st_inv))

    return policy_seq


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

    print create_init_policy(env._traj2pull, env._des_force_traj, ctrl_traj)


