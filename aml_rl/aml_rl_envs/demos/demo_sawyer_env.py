#! /usr/bin/env python

import time
from aml_rl_envs.sawyer.sawyer_env import SawyerEnv


def main():

    environment = SawyerEnv()
    
    motors_ids=[]
    #motors_ids.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
    #motors_ids.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
    #motors_ids.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
    #motors_ids.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
    #motors_ids.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
    
    dv = 0.01 
    motors_ids.append(environment._pb.addUserDebugParameter("posX",-dv,dv,0))
    motors_ids.append(environment._pb.addUserDebugParameter("posY",-dv,dv,0))
    motors_ids.append(environment._pb.addUserDebugParameter("posZ",-dv,dv,0))
    motors_ids.append(environment._pb.addUserDebugParameter("yaw",-dv,dv,0))
    motors_ids.append(environment._pb.addUserDebugParameter("finger_angle",0,0.3,.3))
    
    done = False

    while (not done):
        
      action=[]

      for motor_id in motors_ids:

        action.append(environment._pb.readUserDebugParameter(motor_id))
      
      state, reward, done, info = environment.step2(action)

      obs = environment.get_extended_observation()
      
if __name__=="__main__":

    main()
