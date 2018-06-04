#!/usr/bin/env python

from aml_calib.hand_eye_calib import HandEyeCalib
from aml_math.quaternion_utils import posediff
import numpy as np
import copy
import scipy.io as spio


he_calib = HandEyeCalib()


wHc = spio.loadmat('wHc.mat')
bHg = spio.loadmat('bHg.mat')

transform_gd = np.array([[0.939849549927233, -0.106333628946128, -0.324616670638965, -2.08974078895036e-13],
                         [-0.0469506932684968,0.901082265777532,-0.431099040480069,-9.77368597530024e-14],
                         [0.338346650497956, 0.420409216902265, 0.841889324341649, 50.0000000000000],
                         [0,0,0,1]])



for i in range(wHc['wHc'].shape[2]):

    camera_pose = wHc['wHc'][:,:,i]
    gripper_pose = bHg['bHg'][:,:,i]
    #
    he_calib.add_measurement(gripper_pose, camera_pose)

transform = he_calib.calibrate()

print "Found transform: \n", transform
print "Ground truth: \n", transform_gd

dt, daxis_angle, drot_angle = posediff(transform_gd,transform)

print "Error translation: \n", np.linalg.norm(dt)
print "Error axis angle: \n", daxis_angle*180/np.pi
print "Error rotation angle: \n", drot_angle*180/np.pi
print "Error: ", (transform_gd - transform)






