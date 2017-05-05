#!/usr/bin/env python

import rospy
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_dl.mdn.model.nn_push_fwd_model import NNPushFwdModel
from aml_dl.mdn.model.siamese_push_model import SiamesePushModel
from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_dl.mdn.training.config_exp5 import network_params_inv, fc_network_params_siam, network_params_siam
from aml_services.srv import PredictAction, PredictState, PredictStateResponse, PredictActionResponse

def predict_next_state_from_learned_model(req):

    input_x = np.expand_dims(np.r_[req.state, req.action], 0)

    sess = tf.Session()
    forward_model = NNPushFwdModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()

    next_state = forward_model.run_op('output', input_x)

    return PredictStateResponse(next_state.tolist()[0])


def predict_action_from_learned_model(req):
    sess = tf.Session()

    # inverse_model = MDNPushInverseModel(sess=sess, network_params=network_params_inv)
    # inverse_model.init_model()
    # input_x = np.expand_dims(np.r_[req.curr_state, req.tgt_state], 0)

    inverse_model = SiamesePushModel(sess=sess, network_params=network_params_siam)
    inverse_model.init_model(epoch=48000)

    #############################################THIS IS SPECIFIC TO SIAMESE MODEL####################################
    ##TODO: FIX: For inputs without images, that is for MDNPushInverseModel, there should be a way to give the inputs!
    image_t   = np.asarray(req.curr_state).reshape(1,len(req.curr_state))
    image_t_1 = np.asarray(req.tgt_state).reshape(1,len(req.tgt_state))

    mus = inverse_model.run_op(op_name='mdn_mu',      image_input_t=image_t, image_input_t_1=image_t_1)[0]
    sigma = inverse_model.run_op(op_name='mdn_sigma', image_input_t=image_t, image_input_t_1=image_t_1)[0]
    pis = inverse_model.run_op(op_name='mdn_pi',      image_input_t=image_t, image_input_t_1=image_t_1)[0]

    ###################################################################################################################
    print "MUS: ", mus
    print "SIGMA: ", sigma
    print "PIs:", pis
    #theta = inverse_model.sample_out_max_pi(input_x, 1)[0]   
    pi_idx = np.argmax(pis, axis=0)
    # print "Sigma in prediction \t", sigma[pi_idx]
    action = mus[:,pi_idx]

    return PredictActionResponse(action.tolist(), sigma[pi_idx])


def push_service_server():
    rospy.init_node('push_service_server')
    predict_state  = rospy.Service('predict_state',  PredictState,  predict_next_state_from_learned_model)
    predict_action = rospy.Service('predict_action', PredictAction, predict_action_from_learned_model)
    rospy.spin()


if __name__ == "__main__":
    push_service_server()
