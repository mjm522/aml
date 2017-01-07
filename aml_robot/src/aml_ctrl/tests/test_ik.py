
import rospy
import numpy as np
from aml_robot.baxter_ik import IKBaxter
from aml_robot.baxter_robot import BaxterArm
from store_replay_os_poses import get_saved_test_locations

def test_ik_robot(ik_client, robot_interface, test_poses):

    pos, ori = ik_client.test_pose()

    for k in range(0,4):

        pose = test_poses[k]

        ik_client.configure_ik_service()
    
        soln =  ik_client.ik_servive_request(pos=pose['ee_point'], ori=pose['ee_ori'])

        print "soln \t", np.round(soln, 3)

        robot_interface.move_to_joint_position(soln)

        print "pose done"

   
if __name__ == "__main__":
    
    rospy.init_node('ik_robot', anonymous=True)
    
    limb = 'left'
    
    test_ik_robot(IKBaxter(limb=limb), BaxterArm(limb), get_saved_test_locations())
    
