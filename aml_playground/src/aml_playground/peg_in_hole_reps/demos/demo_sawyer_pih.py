import os
import time
import numpy as np
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_playground.peg_in_hole_reps.controller.sawyer_pih_reps import SawyerPegREPS
#get the experiment params
from aml_playground.peg_in_hole_reps.exp_params.experiment_params import exp_params


def main(joint_space=False):

    ps = SawyerPegREPS(joint_space, exp_params)

    traj = ps.update_dmp_params()[:750,:]

    print ps._env._simulation_reward(traj, end_id = 200)

    test_contexts = np.array([[10,20],[23,32]])

    for i in range (10):

        policy = ps._gpreps.run()

        # test_params = np.array([policy.compute_w(np.array([s]), explore=False) for s in test_contexts])
        
        # mean_reward = np.mean(
        #     np.array([ps._env.reward(p, np.array([s]))[0]
        #               for p, s in zip(test_params, test_contexts)]))

        mean_reward = ps.execute_policy(policy.compute_w(np.array([s])))

        print mean_reward
    
    # for k in range(traj.shape[0]):

    #     if joint_space:

    #         cmd = traj[k, :]

    #     else:

    #         cmd = ps._env._sawyer.inv_kin(ee_pos=traj[k, :].tolist())

    #     ps._env._sawyer.apply_action(cmd)

    #     # time.sleep(0.01)
  
    #     ps._env.simple_step()

  
    raw_input("press enter to exit") 
      
if __name__=="__main__":

    main()