import time
import rospy
import pybullet as pb
from aml_robot.bullet.push_world.push_machine import PushMachine
from aml_robot.bullet.push_world.config import config_push_world


def main():

    rospy.init_node('poke_box', anonymous=True)

    timeStep = 0.01
    
    phys_id = pb.connect(pb.SHARED_MEMORY)
    
    if (phys_id<0):
        phys_id = pb.connect(pb.GUI)
    
    pb.resetSimulation()
    
    pb.setTimeStep(timeStep)
    
    world = pb.loadURDF(config_push_world['world_path'])
    
    pb.setGravity(0., 0.,-10.)

    box = pb.loadURDF(config_push_world['box_path'])

    finger = pb.loadURDF(config_push_world['robot_path'])

    pm = PushMachine(world_id=world, box_id=box, robot_id=finger, config=config_push_world)

    pm.run()


if __name__ == "__main__":    
    main()


        