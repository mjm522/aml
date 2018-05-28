#!/usr/bin/env python

from aml_io.io_tools import load_data
import numpy as np
import sys
import tf

import matplotlib.pyplot as plt


def to_transform_matrix(t, q):
    t_mat = tf.transformations.translation_matrix(t)
    r_mat = tf.transformations.quaternion_matrix(q)
    transform_mat = np.dot(t_mat, r_mat)

    return transform_mat


def inverse_transform(t, q):
    mat = to_transform_matrix(t, q)

    return from_transform_matrix(np.linalg.inv(mat))


def from_transform_matrix(transform):
    t = tf.transformations.translation_from_matrix(transform)
    q = tf.transformations.quaternion_from_matrix(transform)

    return t, q


def multiply_transforms(t1, q1, t2, q2):
    transform_mat1 = to_transform_matrix(t1, q1)
    transform_mat2 = to_transform_matrix(t2, q2)

    transform_mat3 = np.dot(transform_mat1, transform_mat2)

    t = tf.transformations.translation_from_matrix(transform_mat3)
    q = tf.transformations.quaternion_from_matrix(transform_mat3)

    return t, q

if __name__ == '__main__':

    data = load_data(sys.argv[1])


    print "Measurement size: ", len(data['Measurement'])
    print "GroundTruth size: ", len(data['GroundTruth'])

    # print data['GroundTruth']
    # print data['Measurement']


    average_pos_error = []
    count = 0
    nan_count = 0
    correction_t, correction_q = (0,0,0), (0,0,0,1)
    for i in range(len(data['Measurement'])):

        if i == 0 and not np.any(np.isnan(data['GroundTruth'][i][:3])):
            t1 = data['Measurement'][i][:3]
            q1 = data['Measurement'][i][3:]

            t2 = data['GroundTruth'][i][:3]
            q2 = data['GroundTruth'][i][3:]

            invt2, invq2 = inverse_transform(t2,q2)

            correction_t, correction_q = multiply_transforms(t1,q1,invt2,invq2)

        if not (np.any(np.isnan(data['Measurement'][i][:3])) or np.any(np.isnan(data['GroundTruth'][i][:3]))):

            t = data['GroundTruth'][i][:3]
            q = data['GroundTruth'][i][3:]

            tc, qc = multiply_transforms(correction_t,correction_q, t, q)
            transform = np.hstack([tc,qc])

            average_pos_error.append(data['Measurement'][i][:3] - transform[:3])
            count += 1
        else:
            nan_count += 1
            print "Index %d is nan"%(i,), data['Measurement'][i][:3], " or ", data['GroundTruth'][i][:3]
    print "Number of NaNs: ", nan_count
    print "Average Positional Error: ", np.average(average_pos_error)
    print "Average Positional Error Norm: ", np.linalg.norm(np.average(average_pos_error))

    plt.figure()
    plt.plot(average_pos_error)
    plt.show()

