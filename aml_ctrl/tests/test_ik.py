
import rospy
import numpy as np
from aml_robot.baxter_ik import IKBaxter
from aml_robot.baxter_robot import BaxterArm

def test_ik_robot(robot_interface):

    ee_pos, ee_ori = robot_interface._ik_baxter.test_pose()
    

    for k in range(0,100):

        try:
            goal_pos = ee_pos + np.random.randn(3)*0.5
            goal_ori = ee_ori

            print "Goal pos:", goal_pos, " Goal orientation: ", ee_ori
            soln =  robot_interface.inverse_kinematics(pos=goal_pos, ori=goal_ori)

            print "soln \t", np.round(soln, 3)

            robot_interface.exec_position_cmd(soln)

            print "IK: Sucess finding solution"
        except:
            print "IK: Failed to find solution"

        

   
if __name__ == "__main__":
    
    rospy.init_node('ik_robot', anonymous=True)
    
    limb = 'left'
    arm = BaxterArm(limb)

    test_ik_robot(arm)
    
