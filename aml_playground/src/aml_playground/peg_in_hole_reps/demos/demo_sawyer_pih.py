
import numpy as np

from aml_playground.peg_in_hole_reps.controller.sawyer_pih_reps import SawyerPegREPS


def main(joint_space=True):

    ps = SawyerPegREPS(joint_space)

    traj = ps.update_dmp_params()
    
    for k in range(traj.shape[0]):

        if joint_space:

            cmd = traj[k, :]

        else:

            cmd = ps._env._sawyer.inv_kin(ee_pos=traj[k, :].tolist())

        ps._env._sawyer.apply_action(cmd)
                
        ps._env.simple_step()
        
    raw_input("press enter to exit") 
      
if __name__=="__main__":

    main()