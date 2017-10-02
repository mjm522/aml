from aml_robot.box2d.push_world.push_world import PushWorld, STATE
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_robot.box2d.push_world.config import config

from aml_robot.box2d.core.data_manager import DataManager


import tensorflow as tf
import numpy as np

from pygame.locals import (KEYDOWN, K_a,K_s,K_d,K_w,K_p,K_i,K_UP,K_DOWN)
import math
# import matplotlib.backends.backend_agg as agg

from Box2D import (b2Filter,)

# import pylab

import pygame

import matplotlib.pyplot as plt

from aml_dl.mdn.model.tf_ensemble_mdn_model import EnsambleMDN
import os


def get_model(sess):
    EXP_NAME = 'exp_ensemble'

    check_point_dir   = os.environ['AML_DATA'] + '/aml_dl/mdn/tf_check_points/exp_ensemble_pushing/'
    summary_dir = os.environ['AML_DATA'] + '/aml_dl/mdn/summaries/exp_ensemble_pushing/'
    data_dir = os.environ['AML_DATA'] + '/aml_dl/box2d_push_data'

    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    adam_params = {
        'type': 'adam',
        'params': {'learning_rate' : 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
    }

    network_params = {
        'n_ensembles': 5,
        'dim_input': 5, 
        'dim_output': 4,
        'n_hidden': [20,20,10],#[20,20,10],
        'k_mixtures': 1,
        'write_summary': False,
        'load_saved_model': True,
        'model_dir': check_point_dir + '/inv/',
        'model_name':'ensemble_model_inv_10_kernels.ckpt',
        'optimiser': adam_params,
        'summary_dir':summary_dir+'/inv/',
        'device': '/cpu:0',
    }

    ensamble_mdn = EnsambleMDN(network_params,sess)


    return ensamble_mdn



class TestModelPushWorld(PushWorld):

    def __init__(self, config):

        PushWorld.__init__(self,config)
        
        self._box_state = self.get_box_state(self._dynamic_body)

        self._last_push = np.zeros(4) # px, py, f_mag, theta

                # Create a dynamic body
        self._dbody_pred = self._world.CreateStaticBody(position=(16, 12), angle=0)

        self._dbody_pred.linearDamping = 0.6
        self._dbody_pred.angularDamping = 0.05

        self._dbody_pred.awake = False

        # And add a box fixture onto it (with a nonzero density, so it will move)
        self._box_pred = self._dbody_pred.CreatePolygonFixture(box=self._config['box_dim'], density=1, friction=0.3, filter=b2Filter(groupIndex=-8,))
        

        self._box_pred.groupIndex = -8
        self._box.groupIndex = -8

        self._bodies = [self._dynamic_body, self._dbody_pred]

        self._body_idx = 0 

        self._push_action = [0,0,0,0]

        self._app_push = False
        self._predict = False
        self._predicted_state = np.zeros(4)


        ## Tensorflow
        tf.set_random_seed(42)
        self._sess = tf.Session()

        self._model = get_model(self._sess)

        self._model._init_model()
        
        
        # self._sess.run(tf.global_variables_initializer())



    def draw(self, viewer, view_info=True):


        PushWorld.draw(self,viewer,view_info)

        body = self._bodies[self._body_idx]

        ix, iy = self.to_vec(self._push_action[3])

        p = self.get_screen_point(body.position)
        p = (int(p[0]),int(p[1]))

        endpoint = (int(p[0]+ix*50),int(p[1]-iy*50))
        pygame.draw.line(viewer._screen, (255,127,127,255), p, endpoint, 6)



    def handle_event(self,event):
        PushWorld.handle_event(self,event)

        body = self._bodies[self._body_idx]

        ds = 0.5
        if event.type == KEYDOWN:
            if event.key == K_a:
                body.position = [body.position[0]-ds,body.position[1]]
            if event.key == K_d:
                body.position = [body.position[0]+ds,body.position[1]]
            if event.key == K_w:
                body.position = [body.position[0],body.position[1]+ds]
            if event.key == K_s:
                body.position = [body.position[0],body.position[1]-ds]
            if event.key == K_UP:
                self._push_action[3] += 0.2
                self._push_action[3] = self._push_action[3]%(2*np.pi)
                # self._body_idx = (self._body_idx+1)%2
                print "Push angle %f"%(self._push_action[3],)
            if event.key == K_DOWN:
                self._push_action[3] -= 0.2
                self._push_action[3] = self._push_action[3]%(2*np.pi)
                print "Push angle %f"%(self._push_action[3],)
            if event.key == K_p:
                self._app_push = True

            if event.key == K_i:

                state = self.get_box_state(self._bodies[0])
                xt = np.multiply(state['position']-np.array([16.0,12.0]),np.array([1.0/32.0,1.0/24.0]))
                self._predicted_state, unc = self._model.forward(self._sess, [np.r_[xt,state['linear_velocity'],self._push_action[3]]])
                self._predicted_state = self._predicted_state[0]
                unc = np.sqrt(np.sum(unc[0]))
                self._predicted_state[0] *= 16.0
                self._predicted_state[1] *= 12.0
                
                self._predicted_state[0] += 16
                self._predicted_state[1] += 12

                self._dbody_pred.position = (self._predicted_state[0],self._predicted_state[1])

                print "Current state: ", xt
                print "Predicted state: ", self._predicted_state, " Unc:", unc

                self._predict = not self._predict






 
    def update(self, viewer):

        next_state = STATE['NOP']

        body = self._bodies[self._body_idx]

        # px, py, f_mag, theta = self._last_push
        if self._app_push:
            self.apply_push(body=body, px=self._push_action[0], py=self._push_action[1], force_mag=10., theta=self._push_action[3])
            self._app_push = False

        # self.plot()

        return next_state


def main():
    pygame.init()

    push_world = TestModelPushWorld(config = config)
    viewer = Box2DViewer(push_world, config = config)

    viewer.loop()


if __name__ == "__main__":
    main()