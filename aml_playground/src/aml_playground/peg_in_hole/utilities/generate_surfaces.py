import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_io.io_tools import save_data, load_data

np.random.seed(42)

def fun_1(x, y, rtn_sn=True):

    z = x**2 + y

    if rtn_sn:
        surface_normal  = np.array([2.*x, 1., -1.])
        surface_normal /= np.linalg.norm(surface_normal)
        return z, surface_normal
    else:
        return z

def fun_2(x, y, rtn_sn=True):

    tmp = np.sqrt(x**2 + y**2)
    z = np.sin(tmp)

    if rtn_sn: 
        surface_normal  = np.array([(x/tmp)*np.cos(tmp), (y/tmp)*np.cos(tmp), -1.])
        surface_normal /= np.linalg.norm(surface_normal)
        return z, surface_normal
    else:
        return z

def fun_3(x, y, rtn_sn=True):

    tmp = -x*y
    z = np.sin(tmp)

    if rtn_sn:
        surface_normal  = np.array([(y/tmp)*np.cos(tmp), (x/tmp)*np.cos(tmp), 1.])
        surface_normal /= np.linalg.norm(surface_normal)
        return z, surface_normal
    else:
        return z


def get_trajectory(fun, X, Y, rtn_sn=False):
    traj_x    = X.ravel()
    traj_y    = Y.ravel()
    traj_normals = []
    if rtn_sn:
        zs    = []
        for x,y in zip(traj_x, traj_y):
            z, surface_normal = fun(x,y)
            zs.append(z)
            traj_normals.append(surface_normal)
        traj_z    = np.asarray(zs)
    else:
        traj_z    = fun(traj_x, traj_y, False)
    
    return traj_x, traj_y, traj_z, np.asarray(traj_normals)

def mesh_surface(fun, lim, file_path=None):

    fig   = plt.figure()
    ax 	  = fig.add_subplot(111, projection='3d')
    x = y = np.arange(lim[0], lim[1], 0.5)
    X, Y  = np.meshgrid(x, y)
    zs    = []
    surface_normals = []

    for x,y in zip(np.ravel(X), np.ravel(Y)):
        z, surface_normal = fun(x,y)
        zs.append(z)
        surface_normals.append(surface_normal)

    zs    = np.asarray(zs)
    Z     = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.hold(True)

    #np.array([0.]) * np.ones(X.size)

    for k in range(10):
        tmp_X = np.random.randn(1)*np.ones(X.size)
        traj_x, traj_y, traj_z, traj_normals = get_trajectory(fun, tmp_X, Y, True)
        ax.plot(traj_x, traj_y, traj_z)

    if file_path is not None:
        save_data(np.vstack([np.ravel(X), np.ravel(Y), zs]).T, file_path+'surface.pkl')
        save_data(np.asarray(surface_normals), file_path+'surface_normal.pkl')
        save_data(np.vstack([traj_x, traj_y, traj_z]).T, file_path+'surface_trajectory.pkl')

    plt.show()

def traingular_mesh_surface(fun, lim, file_path=None):

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

    Z    = []
    surface_normals = []

    for x,y in zip(X, Y):
        z, surface_normal = fun(x,y)
        Z.append(z)
        surface_normals.append(surface_normal)

    ax  = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if file_path is not None:
        save_data(np.vstack([np.ravel(X), np.ravel(Y), zs]).T, file_path+'surface.pkl')
        save_data(np.asarray(surface_normals), file_path+'surface_normal.pkl')

    plt.show()



def main(fun, lim):

    file_path = os.environ['AML_DATA'] + '/aml_playground/peg_in_hole/surfaces/'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    mesh_surface(fun, lim, file_path)
    # traingular_mesh_surface(fun, lim)


if __name__ == '__main__':
    lim1 = [-5.,5.]
    lim2 = [-3.,3.]
    main(fun=fun_2, lim=lim2)