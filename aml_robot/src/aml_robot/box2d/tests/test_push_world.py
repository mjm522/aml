import pygame
from aml_robot.box2d.push_world.config import config
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.push_world.push_world import PushWorld

def main():
    pygame.init()

    start_positions = [(5, 5), (25, 5), (25, 20)]

    data_file_names = ['data_test_%d.pkl' for d in range(len(start_positions))]
    
    index = 1

    # config['no_samples'] = 2000

    config['box_pos']=start_positions[index]

    push_world = PushWorld(config = config)
    push_world.reset_box()

    viewer = Box2DViewer(push_world, config = config)

    # viewer.threaded_loop()

    sample_id = push_world._data_manager._next_sample_id

    while sample_id < config['no_samples']:
        sample_id = push_world._data_manager._next_sample_id

    push_world.save_samples(data_file_names[index])


if __name__ == "__main__":
    main()