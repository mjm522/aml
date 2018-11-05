import os
import random
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn import preprocessing
from aml_io.io_tools import load_data
from aml_opt.em_gmm_gmr.em_gmm_gmr import EM_GMM_GMR

data_file = os.environ['AML_DATA'] + '/aml_playground/sawyer_spring/new_spring_demos/pre_processed/train_data_total_with_ft1.npy'

Data = np.loadtxt(data_file)

indices = range(Data.shape[0])

random.shuffle(indices)

train_indices = indices[0:8500]
test_indices = indices[8500:]

# plt.plot(train_indices, 'r')
# plt.plot(test_indices, 'g')
# plt.show()
# raw_input()

# train_indices = np.random.choice(range(Data.shape[0]), 9000, replace=False)


X_train = Data[train_indices, :]
X_test = Data[test_indices,:]

# scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
# scaler.transform(X_test)
# scaler.inverse_transform(X)
standard_data_train = X_train.T#scaler.transform(X_train, c#opy=True).T
standard_data_test = X_test.T #scaler.transform(X_test.T #, copy=True).T


em = EM_GMM_GMR(data=standard_data_train)
em.run_em()

types=['front', 'top', 'front_left', 'front_left_top', 'front_right', 'front_right_top', 'left', 'right']
save_path = os.environ['AML_DATA'] + '/aml_playground/sawyer_spring/new_spring_demos/pre_processed/'

for file_name in types:

    print  "Type of file: %s"%file_name
    
    load_data = np.loadtxt(save_path+'train_data_%s.npy'%file_name).T

    mu_out, sigma_out, kp_traj, kd_traj = em.gmr(data_in=load_data, in_=range(9), out=range(9,12))

    # estimated_ = np.vstack([ standard_data_test[0:6, :], mu_out])

    # scale_back = np.tile( np.ones(9), (estimated_.shape[1], 1) ).T

    # estimated_test = np.multiply(estimated_, scale_back) #scaler.inverse_transform(estimated_)
    # estimated_test_org  = np.multiply(standard_data_test, scale_back)

    scatter_x = range(mu_out.shape[1])
    scatter_x_1 = range(load_data.shape[1])

    plt.figure('scaled back', figsize=(10,10))
    plt.cla()
    plt.subplot(1,3,1)
    plt.scatter(scatter_x, mu_out[0, :], c='g')
    plt.scatter(scatter_x_1, load_data[6, :], c='r')
    plt.subplot(1,3,2)
    plt.scatter(scatter_x, mu_out[1, :], c='g')
    plt.scatter(scatter_x_1, load_data[7, :], c='r')
    plt.subplot(1,3,3)
    plt.scatter(scatter_x, mu_out[2, :], c='g')
    plt.scatter(scatter_x_1, load_data[8, :], c='r')


    plt.figure('gains')
    plt.cla()
    plt.subplot(1,3,1)
    plt.scatter(scatter_x, kp_traj[0, :], c='r')
    plt.scatter(scatter_x, kd_traj[0, :],  c='g')
    plt.subplot(1,3,2)
    plt.scatter(scatter_x, kp_traj[1, :],  c='r')
    plt.scatter(scatter_x, kd_traj[1, :],  c='g')
    plt.subplot(1,3,3)
    plt.scatter(scatter_x, kp_traj[2, :],  c='r')
    plt.scatter(scatter_x, kd_traj[2, :],  c='g')
    # plt.figure('standard')
    # plt.plot(standard_data_test[2, :], 'r')
    # plt.plot(estimated_[2, :], 'g')

    plt.show()


    raw_input("Press to continue")