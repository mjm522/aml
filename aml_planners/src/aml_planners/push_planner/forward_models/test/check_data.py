import os
import GPy
import numpy as np
import pandas as pd
import pylab as plt
from sklearn import preprocessing
from controller.utils import get_heatmap
from forward_models.gp_model import GPModel
from config import ensemble_config as config
from forward_models.dropoutNN_model import DropoutNN
from forward_models.ensemble_model import EnsambleModel


ForwardModel = GPModel

if ForwardModel == EnsambleModel:
    from config import ensemble_config as config
elif ForwardModel == DropoutNN:
    from config import dropout_config as config
elif ForwardModel == GPModel:
    config = 3


def combine_files():
    sides = ['0025', '2550', '5075', '75100']
    indices = xrange(1,12)
    total_size = 0

    for side in sides:

        old_data = None
        df_to_save = pd.DataFrame(columns=['xi','yi','thetai','action', 'xf','yf','thetaf'])
        filename_to_save = os.environ['MPPI_DATA_DIR'] + 'baxter_push_data_side_' + side + '_total.csv'

        for  idx in indices:

            filename = os.environ['MPPI_DATA_DIR'] + 'baxter_push_data_side_' + side + '_' + str(idx) + '.csv'

            print "Reading file \n", filename

            try:
                df = pd.DataFrame.from_csv(filename)
            except Exception as e:
                df = None

            if df is None:
                break
            else:
                new_data = np.array(df[['xi', 'yi', 'thetai', 'action', 'xf', 'yf', 'thetaf']])
                if old_data is None:
                    old_data = new_data
                else:
                    old_data = np.vstack([old_data, new_data])

        if old_data is not None:
            tmp = pd.DataFrame(old_data, columns=['xi','yi','thetai', 'action','xf','yf','thetaf'])
            df_to_save = df_to_save.append(tmp, ignore_index=True)
            print "Data of side \t", side
            print old_data.shape
            total_size += old_data.shape[0]
            df_to_save.to_csv(filename_to_save)

    print "Total numnber of pushes \t", total_size


def get_data(side):
    filename = os.environ['MPPI_DATA_DIR'] + 'baxter_push_data_side_' + side  + '_total.csv'
    df = pd.DataFrame.from_csv(filename)
    data = np.array(df[['xi', 'yi', 'thetai', 'action', 'xf', 'yf', 'thetaf']])

    return data


def get_combined_data(sides):
    data = None

    for side in sides:
        if data is None:
            data = get_data(side)
        else:
            data = np.vstack([data, get_data(side)])

    return data

def analysis(sides):

    data = get_combined_data(sides)

    X = data[:, 0:3]
    u = data[:, 3]
    Y = data[:, 4:]

    delta = Y-X

    plt.figure("Plot of deltas against action")

    plt.subplot(311)
    plt.xlabel("action")
    plt.ylabel("delta x")
    plt.scatter(u, delta[:,0])

    plt.subplot(312)
    plt.xlabel("action")
    plt.ylabel("delta y")
    plt.scatter(u, delta[:,1])

    plt.subplot(313)
    plt.xlabel("action")
    plt.ylabel("delta th")
    plt.scatter(u, delta[:,2])


def add_new_dim(u):
    u_new = np.zeros([len(u), 3])
    
    for k in range(len(u)):
        u_new[k,0] =  u[k]
        if 0. <= u[k] <= 0.25:
            u_new[k,1] =  0
            u_new[k,2] =  1
        elif 0.25 < u[k] <= 0.5:
            u_new[k,1] =  -1
            u_new[k,2] =  0
        elif 0.5 <= u[k] <= 0.75:
            u_new[k,1] =  0
            u_new[k,2] =  -1
        elif 0.75 < u[k] <= 1.:
            u_new[k,1] =  1
            u_new[k,2] =  0

    return u_new


def fwd_model_test(sides, test_no=500, show_heat_map=False, add_extra_dim=False):

    data = get_combined_data(sides)

    X = data[:, 0:3]
    u = data[:, 3]
    Y = data[:, 4:]

    if add_extra_dim:
        u = add_new_dim(u)

    if u.ndim == 1:
        u = u[:,None]


    delta = Y-X

    X_test = np.random.multivariate_normal(X.mean(axis=0), np.diag(X.var(axis=0)), test_no)
    u_test = np.random.uniform(0.,1., test_no)

    if u_test.ndim == 1:
        u_test = u_test[:,None]

    if add_extra_dim:
        u_test = add_new_dim(u_test)
    
    X_test = np.hstack([X_test, u_test])

    X_train = np.hstack([X, u])
    Y_train = delta

    fwd_model = ForwardModel(config)
    fwd_model.fit(X_train, Y_train)

    mus_train, sigmas_train = fwd_model.predict(X_train)
    mus_test, sigmas_test = fwd_model.predict(X_test)

    if show_heat_map:
        plt.figure("Heat map")
        heatmap = get_heatmap(model=fwd_model, cmd_dim=1, cost=None, obstacle=None,  r_obs=None)
        # plt.imshow(heatmap, origin='lower', interpolation='none', extent=[0,1.,0,1.])
        # plt.colorbar()

    plt.figure("Plot of predicted deltas against action - test ")
    plt.subplot(311)
    plt.xlabel("action")
    plt.ylabel("delta x")
    plt.scatter(u[:,0], delta[:,0], c='r')
    plt.scatter(u[:,0], mus_train[:,0], c='g')
    plt.scatter(u_test[:,0], mus_test[:,0], c='b')

    plt.subplot(312)
    plt.xlabel("action")
    plt.ylabel("delta y")
    plt.scatter(u[:,0], delta[:,1], c='r')
    plt.scatter(u[:,0], mus_train[:,1], c='g')
    plt.scatter(u_test[:,0], mus_test[:,1], c='b')

    plt.subplot(313)
    plt.xlabel("action")
    plt.ylabel("delta th")
    plt.scatter(u[:,0], delta[:,2], c='r')
    plt.scatter(u[:,0], mus_train[:,2], c='g')
    plt.scatter(u_test[:,0], mus_test[:,2], c='b')

    # subplot_no = (mus_train_list.shape[0]-1)*100+11

    # plt.figure("Plot of predicted deltas against action - train ")
    # for k in range(mus_train_list.shape[0]-1):
    #     plt.subplot(subplot_no)
    #     subplot_no += 1
    #     to_be_plotted = mus_train_list[k,:,:] - mus_train
    #     plt.plot(to_be_plotted[:,0], c='r')
    #     plt.plot(to_be_plotted[:,1], c='g')
    #     plt.plot(to_be_plotted[:,2], c='b')

    
    # subplot_no = (mus_test_list.shape[0]-1)*100+11
    # plt.figure("Plot of predicted deltas against action - test ")
    # for k in range(mus_test_list.shape[0]-1):
    #     plt.subplot(subplot_no)
    #     subplot_no += 1
    #     to_be_plotted = mus_test_list[k,:,:] - mus_test
    #     plt.plot(to_be_plotted[:,0], c='r')
    #     plt.plot(to_be_plotted[:,1], c='g')
    #     plt.plot(to_be_plotted[:,2], c='b')


    
def main():
    sides = ['0025', '2550', '5075', '75100']
    # analysis(sides)
    # get_combined_data(sides)
    fwd_model_test(sides=sides, show_heat_map=False, add_extra_dim=True)
    plt.show()

if __name__ == '__main__':
    main()



            