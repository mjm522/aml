from aml_robot.box2d.push_world.config import config
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.push_world.push_world import PushWorld

def main():
    
    push_world = PushWorld(config = config)

    viewer = Box2DViewer(push_world, config = config)

    sample_id = push_world._data_manager._next_sample_id

    while sample_id < config['no_samples']:
        sample_id = push_world._data_manager._next_sample_id

    push_world.save_samples(config['training_data_file'])


if __name__ == "__main__":
    main()