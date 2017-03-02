#!/usr/bin/env python

import rospy
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from aml_dl.mdn.model.nn_push_fwd_model import NNPushForwardModel
from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_dl.mdn.training.config import network_params_inv, network_params_fwd, check_point_path
from aml_services.srv import PredictAction, PredictState, PredictStateResponse, PredictActionResponse

def predict_next_state_from_learned_model(req):

    input_x = np.expand_dims(np.r_[req.state, req.action], 0)

    sess = tf.Session()
    forward_model = NNPushForwardModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()

    next_state = forward_model.run_op('output', input_x)

    return PredictStateResponse(next_state.tolist()[0])


def predict_action_from_learned_model(req):
    sess = tf.Session()

    inverse_model = MDNPushInverseModel(sess=sess, network_params=network_params_inv)
    inverse_model.init_model()


    input_x = np.expand_dims(np.r_[req.curr_state, req.tgt_state], 0)
    mus = inverse_model.run_op('mu', input_x)[0]
    sigma = inverse_model.run_op('sigma', input_x)[0]
    pis = inverse_model.run_op('pi', input_x)[0]


    print "MUS: ", mus
    print "SIGMA: ", sigma
    print "PIs:", pis
    #theta = inverse_model.sample_out_max_pi(input_x, 1)[0]
    
    pi_idx = inverse_model._max_pi_idx(pis)

    action = mus[:,pi_idx]

    return PredictActionResponse(action.tolist())


def push_service_server():
    rospy.init_node('push_service_server')
    predict_state  = rospy.Service('predict_state',  PredictState,  predict_next_state_from_learned_model)
    predict_action = rospy.Service('predict_action', PredictAction, predict_action_from_learned_model)
    rospy.spin()


if __name__ == "__main__":
    push_service_server()
