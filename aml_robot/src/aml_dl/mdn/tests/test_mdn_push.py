import numpy as np

import tensorflow as tf

from aml_io.io import load_tf_check_point

from aml_robot.box2d.data_manager import DataManager

import matplotlib.pyplot as plt

from aml_dl.mdn.model.tf_model import tf_pushing_model

from aml_dl.mdn.training.config import check_point_path, network_params


def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print 'error with sampling ensemble'
    return -1

def generate_ensemble(out_pi, out_mu, out_sigma, h_test, M=10):
    NTEST  = h_test.size
    result = np.random.rand(NTEST, M) # initially random [0, 1]
    rn  = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
    mu  = 0
    std = 0
    idx = 0

    # transforms result into random ensembles
    for j in range(M):
        for i in range(0, NTEST):
          idx = get_pi_idx(result[i, j], out_pi[i])
          mu = out_mu[i, idx]
          std = out_sigma[i, idx]
          result[i, j] = mu + rn[i, j]*std

    return result

def generate_y_test(session, net):
    
    N_SAMPLES = 1000
    
    x_test = np.float32(np.random.uniform(-5.5, 5.5, (4, N_SAMPLES))).T
    
    h_test = session.run(net['z_hidden'],feed_dict={net['x']: x_test})
    
    NTEST = h_test.size

    out_pi_test, out_sigma_test, out_mu_test = session.run([net['pi'], net['sigma'], net['mu']], feed_dict={net['x']: x_test})

    y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test, h_test)

    plt.figure(figsize=(8, 8))
    plt.plot(h_test, y_test,'bo',alpha=0.1)
    plt.show()

def plot_training_data(session, net):

    data_manager = DataManager.from_file('tests/data_test.pkl')
    data_x = data_manager.pack_data_x(['state_start','state_end'])
    data_y = data_manager.pack_data_y()
    h = session.run(net['z_hidden'],feed_dict={net['x']: data_x})
    plt.figure(figsize=(8, 8))
    plt.plot(h, data_y,'ro', alpha=0.3)
    plt.show()

def main():
    
    check_point_name = check_point_path + 'push_model.ckpt'

    sess = tf.Session()

    net = tf_pushing_model(dim_input= network_params['dim_input'], 
                           n_hidden = network_params['n_hidden'], 
                           n_kernels = network_params['KMIX'])

    
    sess.run(tf.initialize_all_variables())
    

    load_tf_check_point(session=sess, filename=check_point_name)

    

    plot_training_data(session=sess, net=net)

    generate_y_test(session=sess, net=net)


if __name__ == '__main__':
    main()