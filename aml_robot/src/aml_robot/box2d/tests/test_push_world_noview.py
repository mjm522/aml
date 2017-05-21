from aml_robot.box2d.push_world.config import config
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.push_world.push_world import PushWorld
from pygame.locals import *

def main():
    

    push_world = PushWorld(config = config)



    sample_id = push_world._data_manager._next_sample_id

    while sample_id < config['no_samples']:
        sample_id = push_world._data_manager._next_sample_id

        for i in range(config['steps_per_frame']):
            push_world.update()               
            push_world.step()


if __name__ == "__main__":
    main()