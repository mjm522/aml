import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def get_circle(radius, position, steps = 100):
    #for plotting obstacles
    th = np.linspace(0., 2*np.pi, steps)
    x_obs = position[0] + radius*np.cos(th)
    y_obs = position[1] + radius*np.sin(th)


    return x_obs, y_obs


def get_heatmap(model, cmd_dim, div=50, cost=None, obstacle=None, r_obs=None):

    print "Generating heat map"
    ## Heat map
    pxs = np.linspace(0, 1., div)
    pys = np.linspace(0, 1., div)
    X, Y = np.meshgrid(pxs, pys)
    positions = np.vstack([X.ravel(), Y.ravel()]).T


    # XS_Test = np.hstack([positions,np.zeros((positions.shape[0],4)),np.random.randn(positions.shape[0],4)])
    XS_Test = np.hstack([positions, np.zeros((positions.shape[0], 1)), np.random.uniform(0.,1., positions.shape[0])[:,None]])
    mus, sigmas = model.predict(XS_Test)

    print "Sigmas shape: ", sigmas.shape

    noise_level = 0.001
    for i in range(4):
        # x = np.hstack([positions + np.random.randn(positions.shape[0],2)*noise_level,
        #               np.zeros((positions.shape[0],1)), # angular position
        #               np.random.randn(positions.shape[0],2)*noise_level, # linear velocity
        #               np.zeros((positions.shape[0],1))]) # angular velocity
        # u = np.hstack([np.zeros((positions.shape[0],2)), # push position
        #                np.random.randn(positions.shape[0],2)*noise_level]) #push force

        x = np.hstack([positions, np.zeros((positions.shape[0], 1))]) + np.random.randn(positions.shape[0],3)*noise_level
        u = np.random.uniform(0, 1, x.shape[0])[:,None]
        XS_Test = np.hstack([x, u])

        mus_tmp, sigmas_tmp = model.predict(XS_Test)
        sigmas += sigmas_tmp

    sigmas /= 5.0

    sigmas = np.sum(sigmas,axis=1).flatten()

    # plt.figure(1)
    # plt.contour(positions[:,0],positions[:,1],np.reshape(sigmas,(-1,1)))

    print "Pos shape: ", positions.shape, X.shape, Y.shape, sigmas.shape

    fig = plt.figure("Uncertainty plot")
    ax = fig.add_subplot(111, projection='3d')
    Z = griddata((positions[:,0], positions[:,1]), sigmas, (X, Y), method='cubic')

    # s = np.reshape(X.shape,ax)
    ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("sigma")
    
    if obstacle is not None:
        def dist_func(pt):
            return (np.linalg.norm(pt-obstacle)**2-r_obs**2)

        dists = [dist_func(pt) for pt in positions]
        Z_dists = griddata((positions[:,0], positions[:,1]), dists, (X, Y), method='cubic')
        fig = plt.figure("Obstacle cost plot")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z_dists, cmap=cm.coolwarm)

    if cost is not None:
        costs = [cost(np.array([pt[0],pt[1],0,0,0,0]),np.zeros(cmd_dim),np.zeros(cmd_dim),unc,0) for pt,unc in zip(positions,sigmas)]
        Z_costs = griddata((positions[:,0], positions[:,1]), costs, (X, Y), method='cubic')
        fig = plt.figure("Cost plot")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z_costs, cmap=cm.coolwarm)


    heatmap = np.reshape(sigmas,(div,div))


    return heatmap

    # plt.figure(2)
    # plt.imshow(sigmas, origin='lower', interpolation='nearest', extent=[0,7,0,7])
    # plt.colorbar()




