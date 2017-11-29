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

    peg = pb.loadURDF(config_pih_world['peg_path'])

    pb.setGravity(0., 0.,-10.)

    hole = pb.loadURDF(config_pih_world['hole_path'], useFixedBase=True)

    manipulator = pb.loadURDF(config_pih_world['robot_path'], useFixedBase=True, globalScaling=1.5)

    motors = [n for n in range(pb.getNumJoints(manipulator))]

    velocity_array = [0, -0.05, -0.05]

    pb.setJointMotorControlArray(manipulator, motors,controlMode=pb.VELOCITY_CONTROL, targetVelocities=velocity_array, forces=[500 for n in range(len(velocity_array))])

    # print "Motors: ", motors
    # print "INFO", pb.getJointInfo(manipulator,2)

    pm = PIHWorld(world_id=world, peg_id=peg, hole_id=hole, robot_id=manipulator, config=config_pih_world)

    pm.run()


if __name__ == "__main__":    
    main()


        