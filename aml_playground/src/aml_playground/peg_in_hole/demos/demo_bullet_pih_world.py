import numpy as np
from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config


def main():

    gains = np.array([1,1,1])

    velocity_array = gains*np.array([0, -0.05, -0.05])

    pm = PIHWorld(config=pih_world_config)

    while True:

        jnt_pos, jnt_vel, _, _ =  pm._manipulator.get_jnt_state()

        pm.update(velocity_array)

        print "Joint position is \t", jnt_pos
        print  "Joint vel is \t",  jnt_vel

        pm.step()


if __name__ == "__main__":    
    main()