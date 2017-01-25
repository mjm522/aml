from aml_robot.box2d.push_world import PushWorld
from aml_robot.box2d.pygame_viewer import PyGameViewer
from aml_robot.box2d.config import config

from aml_robot.box2d.data_manager import DataManager

from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_dl.mdn.training.config import network_params, check_point_path

import tensorflow as tf
import numpy as np

# import matplotlib.backends.backend_agg as agg



# import pylab

import pygame

import matplotlib.pyplot as plt

config['record_training_data'] = True
network_params['load_saved_model'] = True
network_params['model_path'] = check_point_path + 'push_model_pi_div_two.ckpt'

train_on_fly = False


class TestModelPushWorld(PushWorld):

    def __init__(self, config):

        PushWorld.__init__(self,config)
        
        self._box_state = self.get_box_state(self._dynamic_body)

        self._sess = tf.Session()

        self._inverse_model = MDNPushInverseModel(sess=self._sess, network_params=network_params)

        self._inverse_model.init_model()

        self._change_after = 100

        self._loss = []


    def draw(self, screen):
        PushWorld.draw(self,screen)


        body = self._dynamic_body

        # tgt = np.random.randn(2)
        state = self._box_state['linear_velocity']
        # px, py, tgt_x, tgt_y, theta = self._last_push
        input_x = np.expand_dims(np.r_[np.zeros(2), state],0)


        # theta = self._inverse_model.expected_out(input_x,1)

        # ix = np.cos(theta)
        # iy = np.sin(theta)

        ix, iy = self._inverse_model.expected_out2(input_x,5)

        # print "IXIY:", ix, iy

        px, py = self._box_state['position']

        p = self.get_screen_point((px,py))
        p = (int(p[0]),int(p[1]))

        endpoint = (int(p[0]+ix*25),int(p[1]-iy*25))
        # print "DIR: ", direction
        pygame.draw.line(screen,(255,127,127,255),p,endpoint, 4)


    def is_time_to_train(self, sample):

        return train_on_fly and self._data_manager.size() > 1 and sample is not None and not sample['stale'] and sample['sample_id']%100 == 0
    

    def plot(self):
        # fig = pylab.figure(figsize=[4, 4], # Inches
        #            dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
        #            )
        # ax = fig.gca()
        # ax.plot([1, 2, 4])
         
        # canvas = agg.FigureCanvasAgg(fig)
        # canvas.draw()
        # renderer = canvas.get_renderer()
        # raw_data = renderer.tostring_rgb()

        # plt.show(block=False)
        pass
 
    def update(self, viewer):

        next_state = PushWorld.update(self, viewer)

        self._box_state = self.get_box_state(self._dynamic_body)

        latest_sample = self._data_manager.get_last()
        if self.is_time_to_train(latest_sample):
            latest_sample['stale'] = True

            batch_ids = self._data_manager.get_last_ids(100)
            data_x, data_y = self._data_manager.pack_data(['state_start','state_end'],batch_ids)


            # print "DATA_X: ", data_x
            # print "DATA_Y: ", data_y
            self._loss = np.r_[self._loss,self._inverse_model.train(data_x, data_y, epochs=100)]


            # print "SHOULD TRAIN!!!"
            # plt.ion()
            plt.plot(self._loss)
            # plt.plot(range(len(self._loss)),self._loss)
            plt.show(block=False)
            plt.draw()


        # self.plot()

        return next_state



push_world = TestModelPushWorld(config = config)
viewer = PyGameViewer(push_world, config = config)


viewer.loop()


push_world._inverse_model.save_model()
push_world.save_samples('data_test_pi_div_2.pkl')