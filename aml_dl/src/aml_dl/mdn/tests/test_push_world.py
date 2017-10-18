from aml_robot.box2d.push_world.push_world import PushWorld
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.config import config

from aml_robot.box2d.core.data_manager import DataManager

from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_dl.mdn.training.config import network_params_inv, check_point_path

import tensorflow as tf
import numpy as np

from pygame.locals import (KEYDOWN, K_m)
import math
# import matplotlib.backends.backend_agg as agg



# import pylab

import pygame

import matplotlib.pyplot as plt



config['record_training_data'] = True
config['steps_per_frame'] = 1

network_params['load_saved_model'] = False
network_params['model_path'] = check_point_path + 'push_model_k5_h10_360_damp.ckpt'
network_params['dim_input'] = 6
network_params['dim_output'] = 3
network_params['k_mixtures'] = 5
network_params['n_hidden'] = 10

train_on_fly = False

show_mixture = True



class TestModelPushWorld(PushWorld):

    def __init__(self, config):

        PushWorld.__init__(self,config)
        
        self._box_state = self.get_box_state(self._dynamic_body)

        self._sess = tf.Session()

        self._inverse_model = MDNPushInverseModel(sess=self._sess, network_params=network_params)

        self._inverse_model.init_model()

        self._change_after = 100

        self._loss = []


    def draw_prediction(self, screen, box_center, theta, sigma = 1, colour = (255,127,127,255), radius = 25, sigma_scale = 100, draw_at_circle_endpoint = False):

        ix, iy = self.to_vec(theta)

        p = self.get_screen_point(box_center)
        p = (int(p[0]),int(p[1]))

        endpoint = (int(p[0]+ix*radius),int(p[1]-iy*radius))

        pygame.draw.line(screen, colour,p, endpoint, 6)

        if not draw_at_circle_endpoint:
            pygame.draw.circle(screen, (127,255,255,255), p, min(int(sigma*sigma_scale+2),50),2)
        else:
            pygame.draw.circle(screen, (127,255,255,255), endpoint, min(int(sigma*sigma_scale+2),50),2)

    def draw(self, viewer):


        PushWorld.draw(self,viewer)

        screen = viewer._screen

        body = self._dynamic_body

        # tgt = np.random.randn(2)
        curr_vel = self._box_state['linear_velocity']
        curr_pos = self._box_state['position']
        curr_ang = self._box_state['angle']
        # px, py, tgt_x, tgt_y, theta = self._last_push
        input_x = np.expand_dims(np.r_[np.zeros(3), np.multiply(curr_pos - np.array([16.0,12.0]),[1./32.0,1./24.0]), curr_ang],0)

        # print "INPUT_X: ", input_x
        # theta = self._inverse_model.expected_out(input_x,1)

        # ix = np.cos(theta)
        # iy = np.sin(theta)

        # ix, iy = self._inverse_model.expected_max_pi_out2(input_x,20)
        # theta = np.arctan2(iy,ix)

  
        # prediction = self._inverse_model.expected_max_pi_out2(input_x, 20)

        mus = self._inverse_model.run_op('mu', input_x)
        # print mus
        sigmas = self._inverse_model.run_op('sigma', input_x)[0]
        pis = self._inverse_model.run_op('pi', input_x)[0]
        
        # if show_mixture:
        #     for i in range(len(pis)):
        #         self.draw_prediction(screen, curr_pos, mus[i], sigmas[i], (255,255,127,10), 80, sigma_scale = 20, draw_at_circle_endpoint = True)


        max_idx = np.argmax(pis)
        sigma = sigmas[max_idx]
        mu = mus[max_idx]

        px = mu[0] 
        py = mu[1]
        theta = mu[2]


        predicted_push_point = self.get_point(body,(float(px),float(py)))


        
        gt_px, gt_py, gt_ix, gt_iy, gt_theta = self._last_push

        
        self.draw_prediction(screen, predicted_push_point, theta, sigma, colour = (255,127,127,255), radius = 60, sigma_scale = 20)

        p = self.get_screen_point2(body,(gt_px,gt_py))
        self.draw_prediction(screen, p, gt_theta, 1, colour = (127,255,127,255), radius = 60, sigma_scale = 20)
        
        

        _, _, _, _, theta_gt = self._last_push
        label = viewer.create_text_surface("Predicted angle %0.2f uncertainty %0.2f error %0.2f"%(theta*180/np.pi, sigma, (theta-theta_gt)*180/np.pi))
        
        screen.blit(label, (0, 30))


    def handle_event(self,event):
        PushWorld.handle_event(self,event)
        global show_mixture

        if (event.type == KEYDOWN and event.key == K_m):
            show_mixture = not show_mixture



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


            print "DATA_X: ", len(data_x), len(data_x[0])
            print "DATA_Y: ", len(data_y), len(data_y[0])
            # data_y = [np.array(data_y)[0]]
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
viewer = Box2DViewer(push_world, config = config)


viewer.loop()


# push_world._inverse_model.save_model()
push_world.save_samples('data_sim_push.pkl')
