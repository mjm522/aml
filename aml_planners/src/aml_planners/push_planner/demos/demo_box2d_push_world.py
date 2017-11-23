import pylab
import numpy as np
import pygame as pg

from aml_planners.push_planner.box2d_viewer.box2d_viewer import Box2DViewer
from aml_planners.push_planner.push_worlds.box2d_push_world import Box2DPushWorld
from aml_planners.push_planner.push_worlds.config import push_world_config as config


def dynamics(x,u,dt, box):


    mass = box._dyn_body.mass

    A = np.array([[1,0,0,0,1,0,0],
                  [0,1,0,0,0,1,0],
                  [0,0,1,0,0,0,1],
                  [0,0,0,1,0,0,0],
                  [0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,1]])
    B = np.array([[0,0,0],
                  [0,0,0],
                  [dt/m,0,0],
                  [0,dt/m,0]])

    pos = x[:2]
    ang = x[2]
    vel = x[3:5]
    vel_ang = x[6]
    
    force_mag = u[2]
    force_angle = u[3]

    mass = box._dyn_body.mass

    force = np.array([np.cos(force_angle + ang), np.sin(force_angle + ang)])

    px, py = self._dyn_body.transform*(u[0],u[1])
    # torque = np.cross(force, np.array([px,py]) - pos)




    acc = (force_mag*force)/mass

    vel_next = vel + acc*dt
    pos_next = pos + vel*dt

    x_next = np.r_[pos_next, ang, vel_next, vel_ang]

    return x_next


def main():

    # dt = 
    world = Box2DPushWorld(config)

    body_params = {'mass': world._box._dyn_body.mass }

    viewer = Box2DViewer(world, config, is_thread_loop=False)

    
    action = world.sample_push_action3()
    world.update(action)
    pc = 0

    while viewer._running:

        viewer.handle_events()

        viewer.clear_screen(color=(255,255,255,255))


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
            action = world.sample_push_action3()

            world.update(action)
            # world.reset()
            pc = 0






if __name__ == '__main__':
    main()