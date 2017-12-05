import pylab
import numpy as np
import pygame as pg

from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_playground.peg_in_hole.pih_worlds.box2d.box2d_pih_world import Box2DPIHWorld
from aml_playground.peg_in_hole.pih_worlds.box2d.config import pih_world_config as config


def main():

    # dt = 
    world = Box2DPIHWorld(config)

    body_params = {'mass': world._box._dyn_body.mass }

    viewer = Box2DViewer(world, config, is_thread_loop=False)

    
    action = world.sample_action()
    world.update(action)
    pc = 0

    while viewer._running:

        viewer.handle_events()

        viewer.clear_screen(color=(255,255,255,255))

        # print world._box.get_vertices_phys()

        world._manipulator.set_joint_speed([-0.,0.,0.])

        state0 = world.pack_box_state()


        ## Controller selects push action (random sampling for simplicity now)
        # action = world.sample_push_action()

        ## Send action to world
        # world.update(action)

        ## Step the world for certain number of time steps
        for i in range(viewer._steps_per_frame):       
            world.step()
            pass

        statef = world.pack_box_state()
        # print "State0: ", state0, " Action: ", action, " StateF: ", statef, " SDiff: ", statef-state0
    
    
        viewer.draw()

        pc += 1

        if pc >= 5:
            action = world.sample_action()

            # world.update(action)
            # world.reset()
            pc = 0






if __name__ == '__main__':
    main()