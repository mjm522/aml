import time
import rospy
import pybullet as pb
from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.bullet.config import config_pih_world


def main():

    rospy.init_node('poke_box', anonymous=True)

    timeStep = 0.01
    
    phys_id = pb.connect(pb.SHARED_MEMORY)
    
    if (phys_id<0):
        phys_id = pb.connect(pb.GUI)
    
    pb.resetSimulation()
    
    pb.setTimeStep(timeStep)
    
    world = pb.loadURDF(config_pih_world['world_path'])

    hole = pb.loadURDF(config_pih_world['hole_path'], [0,0,1], [0, 0, -0.707, 0.707], useFixedBase=True)

    manipulator = pb.loadURDF(config_pih_world['robot_path'], [0,-2.5,0], useFixedBase=True, globalScaling=1.5)

    peg = pb.loadURDF(config_pih_world['peg_path'], [0,-1.5,3])

    # pb.setGravity(0., 0.,-10.)

    pm = PIHWorld(world_id=world, peg_id=peg, hole_id=hole, robot_id=manipulator, config=config_pih_world)

    pm.run()


if __name__ == "__main__":    
    main()


        