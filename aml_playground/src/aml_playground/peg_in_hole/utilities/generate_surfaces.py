import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_io.io_tools import save_data, load_data

np.random.seed(42)

def fun_1(x, y):
    return x**2 + y

def fun_2(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

def fun_3(x, y):
    return np.sin(-x*y)


def mesh_surface(fun, lim, file_name=None):
    fig   = plt.figure()
    ax 	  = fig.add_subplot(111, projection='3d')
    x = y = np.arange(lim[0], lim[1], 0.05)
    X, Y  = np.meshgrid(x, y)
    zs    = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z     = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if file_name is not None:
        save_data(np.hstack([]),file_name)

    plt.show()

def traingular_mesh_surface(fun, lim, file_name=None):
    fig = plt.figure()
    
    n_radii  = 8
    n_angles = 36

    # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
    radii  = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Repeat all angles for each radius.
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

    # Convert polar (radii, angles) coords to cartesian (x, y) coords.
    # (0, 0) is manually added at this stage,  so there will be no duplicate
    # points in the (x, y) plane.
    X = np.append(0, (radii*np.cos(angles)).flatten())
    Y = np.append(0, (radii*np.sin(angles)).flatten())
    # Compute z to make the pringle surface.
    Z = fun(X,Y)

    ax  = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if file_name is not None:
        save_data(np.hstack([]),file_name)

    plt.show()



def main(fun, lim):
    file_path = os.environ['AML_DATA'] + '/aml_playground/peg_in_hole/surfaces/'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = file_path + 'surface.pkl'
    mesh_surface(fun, lim)
    # traingular_mesh_surface(fun, lim)


if __name__ == '__main__':
    lim1 = [-5.,5.]
    lim2 = [-3.,3.]
    main(fun=fun_3, lim=lim2)