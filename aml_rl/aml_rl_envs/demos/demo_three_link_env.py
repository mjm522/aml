
from bullet_envs.threelink.threeLinkGymEnv import ThreeLinkGymEnv
import time

def main():

	environment = ThreeLinkGymEnv(renders=True, isDiscrete=False, maxSteps = 10000000)
	 
	motorsIds=[]
	
	dv = 0.01 
	motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
	
	done = False
	while (not done):
	    
	  action=[]
	  for motorId in motorsIds:
	    action.append(environment._p.readUserDebugParameter(motorId))

	  state, reward, done, info = environment.step2(action)
	  obs = environment.getExtendedObservation()
	  
if __name__=="__main__":
    main()
