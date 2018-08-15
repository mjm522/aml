import os
import numpy as np
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data
from demo_analyze_spring_test import xyz_plot
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_rl_envs.point_mass.point_mass_env import PointMassEnv
# from aml_playground.var_imp_reps.exp_params.experiment_var_imp_params import exp_params
from aml_playground.var_imp_reps.exp_params.experiment_point_mass_params import exp_params


env_params = exp_params['env_params']
env_params['renders'] = True

env = PointMassEnv(env_params)

file_path = exp_params['param_file_name']

data = load_data(file_path)

class DummyPolicy():
    """
    this is a dirty hack policy
    when a context is called it
    simply returns from a saved list of 
    values
    """

    def __init__(self, t):

        self._count = 0
        # Kp = 0.6;
        # Kd = 0.8;
        # self._w_list = np.tile( np.hstack([np.ones(3)*Kp, np.ones(3)*Kd]) , (200, 1) )
        tmp = np.ones(6) #np.array([ 1./1000.,  1./1000., 1./1000.,  1./np.sqrt(1000.),  1./np.sqrt(1000.),  1./np.sqrt(1000.)])
        self._w_list = np.multiply(data[-1]['params'][t], tmp)

    def compute_w(self, context, explore):
        w = self._w_list
        self._count += 1
        return w

policy = [DummyPolicy(k) for k in range(100)]

traj_draw, reward = env.execute_policy(policy=policy, explore=False)

ee_traj=traj_draw['ee_traj']
w_list = np.asarray(traj_draw['params'])
req_traj = traj_draw['traj']

plotdata = [ee_traj, w_list[:,:3], w_list[:,3:]]
labels = [['ee_x','ee_y','ee_z'], ['delta kp_x','delta kp_y','delta kp_z'], ['delta kd_x','delta kd_y','delta kd_z']]
axis_labels =[ [ ['steps', 'm'], ['steps', 'm'] , ['steps', 'm']  ],
             [ ['steps', 'N/m'], ['steps', 'N/m'] , ['steps', 'N/m']  ],
             [ ['steps', 'Ns/m'], ['steps', 'Ns/m'] , ['steps', 'Ns/m']  ] ]

# legend = [[]]

mean_reward = [data[i]['mean_reward']  for i in range(len(data))]

# plt.figure()
# xyz_plot(data=plotdata, labels=labels, axis_labels=axis_labels, multiplot={ 1 : [req_traj] } ,title="Test Impedance Learning with 2 external stiffness")

# param_traj_kp = np.zeros(len(w_list))
# param_traj_kd = np.zeros(len(w_list))
# param_traj = np.zeros(len(w_list))

# for k in range(len(w_list)):
#     param_traj[k] = np.linalg.norm(w_list[k,:])
#     param_traj_kp[k] = np.linalg.norm(w_list[k,:3])
#     param_traj_kd[k] = np.linalg.norm(w_list[k,3:])

# plt.figure()
# plt.plot(param_traj, 'r')
# plt.plot(param_traj_kp, 'g')
# plt.figure()
# plt.plot(param_traj, 'r')
# plt.plot(param_traj_kd, 'b')
# plt.title('Mean Reward')
# plt.xlabel('episodes')
# plt.ylabel('magnitude')
plt.show()


