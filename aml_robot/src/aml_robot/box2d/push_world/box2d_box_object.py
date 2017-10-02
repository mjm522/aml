import time
import numpy as np
from aml_robot.box2d.push_world.config import config
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.push_world.push_world import PushWorld


class Box2DBoxObject(PushWorld):

    def __init__(self, config=config):
        PushWorld.__init__(self, config=config)
        self._push_action = (0,0,0,0)

    def new_push(self, push):
        self._push_action = push

    def update(self, viewer):
        next_state = self._current_state

        body = self._dynamic_body

        self.apply_push(body=body, 
                        px=self._push_action[0], 
                        py=self._push_action[1], 
                        force_mag=self._push_action[2], 
                        theta=self._push_action[3])

    def get_pose(self):
        state   = self.get_box_state(self._dynamic_body)
        pose    = None 
        box_pos = state['position']
        box_q   = state['angle']
        pose = np.array([[np.cos(box_q), -np.sin(box_q), box_pos[0]],
                         [np.sin(box_q),  np.cos(box_q), box_pos[1]],
                         [0.,0.,0.,1.]])
        return pose, box_pos, box_q

    def get_curr_image(self):
        state   = self.get_box_state(self._dynamic_body)
        return state['image_rgb']

def main():
    
    box_object = Box2DBoxObject(config = config)
    box_object.reset_box()

    viewer = Box2DViewer(box_object, config = config)

    while True:
        px, py, f_mag, theta = box_object.generate_random_push()
        box_object.new_push(push=(px,py,f_mag, theta))
        time.sleep(2.)
        box_object.reset_box()

    # viewer.loop()


if __name__ == "__main__":
    main()