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

    if len(sys.argv) > 1:
        data_list = [sys.argv[1]]
    else:
        data_list = ["thumb_experiment1.pkl",
                     "index_experiment1.pkl",
                     "middle_experiment1.pkl",
                     "ring_experiment1.pkl",
                     "little_experiment1.pkl"]

    list_pos_error = []
    list_rot_error = []
    list_avg_pos_error = []
    list_avg_rot_error = []
    list_norm_pos_error = []
    list_norm_rot_error = []
    for data_file in data_list:

        data = load_data(data_file)

        print "Measurement size: ", len(data['Measurement'])
        print "GroundTruth size: ", len(data['GroundTruth'])

        # print data['GroundTruth']
        # print data['Measurement']


        average_pos_error = []
        average_rot_error = []
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

                t2, q2 = inverse_transform(data['Measurement'][i][:3],data['Measurement'][i][3:])


                tc, qc = multiply_transforms(correction_t,correction_q, t, q)
                transform = np.hstack([tc,qc])

                terr, qerr = multiply_transforms(t2,q2, tc, qc)
                average_pos_error.append(terr)
                average_rot_error.append(qerr[:3])
                count += 1
            else:
                nan_count += 1
                print "Index %d is nan"%(i,), data['Measurement'][i][:3], " or ", data['GroundTruth'][i][:3]

        print "Number of NaNs: ", nan_count

        def diag(avg_error, name = "Positional"):

            print "Average %s Error: "%(name,), np.average(avg_error)
            print "Average %s Error Norm: "%(name,), np.linalg.norm(np.abs(avg_error))


        diag(average_pos_error)
        diag(average_rot_error,"Orientation")

        plt.figure()
        plt.plot(average_pos_error)

        plt.figure()

        plt.plot(average_rot_error)
        plt.show(False)

        list_pos_error.append(average_pos_error)
        list_rot_error.append(average_rot_error)
        list_avg_pos_error.append(np.average(np.abs(average_pos_error)))
        list_avg_rot_error.append(np.average(np.abs(average_rot_error)))
        list_norm_pos_error.append(np.linalg.norm(np.abs(average_pos_error)))
        list_norm_rot_error.append(np.linalg.norm(np.abs(average_rot_error)))


    plt.figure()
    x = range(len(list_avg_pos_error))
    plt.bar(x,list_avg_pos_error)
    plt.title("Average position error per finger")

    plt.figure()
    plt.bar(x, list_avg_rot_error)
    plt.title("Average rotation error per finger")

    plt.figure()
    x = range(len(list_norm_pos_error))
    plt.bar(x,list_norm_pos_error)
    plt.title("Norm L2 position error per finger")

    plt.figure()
    plt.bar(x, list_norm_rot_error)
    plt.title("Norm L2 rotation error per finger")

    plt.show()

