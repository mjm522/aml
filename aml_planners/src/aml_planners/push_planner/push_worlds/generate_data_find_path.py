import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from push_world.config import config
from aml_io.io_tools import save_data
from push_world.push_world import PushWorld
from push_world.data_collector import DataCollector
from push_world.data_vis import visualise_push_data
from forward_models.ensemble_model import EnsambleModel
from planner.low_cost_path_finder import LowCostPathFinder
from controller.experiment_params import experiment_config as econfig


def get_choice():
    print "Choose type of data set \n (1) Top right corner \n (2) Top left corner\
    \n (3) Bottom right corner \n (4) Bottom left corner \n (5) u shape \
    \n (6) n shape \n (7) c shape \n (8) reverse c shape \n (9) o shape"

    index = raw_input("Choose an option :=")
    index =  int(index)

    if index == 1:
        name = "top_left_shape"
    elif index == 2:
        name = "top_right_shape"
    elif index == 3:
        name = "bottom_right_shape"
    elif index == 4:
        name = "bottom_left_shape"
    elif index == 5:
        name = "u_shape"
    elif index == 6:
        name = "n_shape"
    elif index == 7:
        name = "c_shape."
    elif index == 8:
        name = "rev_c_shape"
    elif index == 9:
        name = "o_shape.csv"

    push_data_filename        = '../data/' + 'se_push_data_' + name + '.csv'
    push_data_image_filename  = '../data/' + 'se_push_data_' + name + '.png'
    heatmap_data_filename     = '../data/' + 'se_heatmap_'   + name + '.pkl'
    heatmap_image_filename    = '../data/' + 'se_heatmap_'   + name + '.png'
    trajectory_image_filename = '../data/' + 'se_traj_'      + name + '.png'

    choice = {
    'index': index,
    'push_data_file': push_data_filename,
    'push_data_image_file':push_data_image_filename,
    'heatmap_data_file':heatmap_data_filename,
    'heatmap_image_file':heatmap_image_filename,
    'traj_image_file':trajectory_image_filename,
    }

    return choice


def gather_data(world, index, data_file, data_image_file):

    xs = np.linspace(0, 8, 5)
    ys = np.linspace(0, 8, 5)

    x0s = []

    for x in xs:
        for y in ys:
            if index == 1:
                #top left corner
                if x < 5 and y > 5:
                    continue
            elif index == 2:
                #top right corner
                if x > 5 and y > 5:
                    continue
            elif index == 3:
                # bottom left corner
                if x < 5 and y < 5:
                    continue
            elif index == 4:
                #bottom right corner
                if x > 5 and y < 5:
                    continue
            elif index == 5:
                # u shape
                if y > 3:
                    if x > 3 and x < 6:
                        continue
            elif index == 6:         
                # n shape
                if y < 6:
                    if x > 3 and x < 6:
                        continue
            elif index == 6:
                # c shape
                if x > 3:
                    if y > 3 and y < 6:
                        continue
            elif index == 8:
                # rev_c shape
                if x < 6:
                    if y > 3 and y < 6:
                        continue
            elif index == 9:
                # o shape
                if x > 3 and x < 6:
                    if y > 3 and y < 6:
                        continue  

            x0 = np.array([x,y,0.,0.,0.,0.])
            x0s.append(x0.copy())

    dc = DataCollector(world=world, config=config, initial_box_states = x0s, 
                      pushes_per_location = 50, n_seq_pushes = 1, noise = 0.1)#0.0001

    dc.collect_data()

    dc.save_csv(data_file)

    visualise_push_data(data_file, data_image_file)


def fit_data_gen_heat_map(data_file, heatmap_data_file=None, heatmap_image_file=None):
    ## Get the data
    df = pd.DataFrame.from_csv(data_file)

    X = np.array(df[['xi','yi','thetai','dxi','dyi','dthetai','a_px','a_py','a_fx', 'a_fy']])

    actions = np.array(df[['a_px','a_py','a_fx', 'a_fy']])

    y = np.array(df[['xf','yf','thetaf','dxf','dyf','dthetaf']])

    print "X: ", X.shape

    print "y: ", y.shape

    em = EnsambleModel(config=econfig)

    print "Fitting model"
    # Fitting model
    em.fit(X,y)

    print "Generating heat map"
    ## Heat map
    pxs = np.linspace(0, 8, 100)
    pys = np.linspace(0, 8, 100)
    X, Y = np.meshgrid(pxs, pys)
    positions = np.vstack([X.ravel(), Y.ravel()]).T


    XS_Test = np.hstack([positions,np.zeros((positions.shape[0],4)),np.random.randn(positions.shape[0],4)])
    mus, sigmas = em.predict(XS_Test)

    noise_level = 0.001
    for i in range(4):
        x = np.hstack([positions + np.random.randn(positions.shape[0],2)*noise_level,
                      np.zeros((positions.shape[0],1)), # angular position
                      np.random.randn(positions.shape[0],2)*noise_level, # linear velocity
                      np.zeros((positions.shape[0],1))]) # angular velocity
        u = np.hstack([np.zeros((positions.shape[0],2)), # push position
                       np.random.randn(positions.shape[0],2)*noise_level]) #push force
        XS_Test = np.hstack([x,u])


        mus_tmp, sigmas_tmp = em.predict(XS_Test)

        sigmas += sigmas_tmp

    sigmas /= 5.0

    sigmas = np.sum(sigmas,axis=1)

    sigmas = np.reshape(sigmas,(100,100))

    plt.figure(2)
    plt.imshow(sigmas, origin='lower', interpolation='nearest', extent=[0,7,0,7])
    plt.colorbar()
    plt.show(False)

    if heatmap_data_file is not None:
        save_data(sigmas, heatmap_data_file)

    if heatmap_image_file is not None:
        plt.savefig(heatmap_image_file)


def find_low_cost_traj(sg, state_constraints, heatmap_data_file):
    lcpf = LowCostPathFinder(sg=sg, state_constraints=state_constraints, heatmap_data_file=heatmap_data_file)
    lcpf.run()


def main():
 
    world = PushWorld(config)

    choice = get_choice()

    gather_data(world=world,
                index=choice['index'], 
                data_file=choice['push_data_file'], 
                data_image_file=choice['push_data_image_file'])


    plt.show()

    fit_data_gen_heat_map(data_file=choice['push_data_file'],
                          heatmap_data_file=choice['heatmap_data_file'],
                          heatmap_image_file=choice['heatmap_image_file'])

    # sg ={
    # 'start':np.array([0.75, 6.75]),
    # 'goal':np.array([6.5, 0.75]),
    # 'obstacle':np.array([3.5,3.5]),
    # 'r_obs':1.0,
    # }

    # state_constraints={
    # 'x':{'min':0., 'max':7.},
    # 'y':{'min':0., 'max':7.},
    # }

    # find_low_cost_traj(sg=sg, 
    #                    state_constraints=state_constraints, 
    #                    heatmap_data_file=choice['heatmap_data_file'])

if __name__ == '__main__':
    main()