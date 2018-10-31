import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_opt.em_gmm_gmr.em_gmm_gmr import EM_GMM_GMR, R

np.random.seed(123)

means = [ [0, 0], [5, 8], [5, 2] ]
covs =  [ [[1, 0], [0, 10]],  [[5, 0], [0, 5]],  [[10, 0], [0, 1]] ] # diagonal covariance
colors = ['r', 'g', 'b']
data = None
data_test = None

for mean, cov in zip(means, covs):
    
    x, y = np.random.multivariate_normal(mean, cov, 200).T

    if data is None:
        data = np.vstack([x, y]).T
    else:
        data = np.vstack([data, np.vstack([x, y]).T])

for mean, cov in zip(means, covs):
    
    x, y = np.random.multivariate_normal(mean, cov, 50).T

    if data_test is None:
        data_test = np.vstack([x, y]).T
    else:
        data_test = np.vstack([data_test, np.vstack([x, y]).T])


# plt.figure('input_data')
# for i, datum in enumerate(data):
#     plt.scatter(datum[:,0], datum[:,1], color=colors[i], marker='x')

# print data.shape

em = EM_GMM_GMR(data=data.T)
em.run_em()
r = em.estimate_attractor_path(DataIn=data_test[:,0], r=R())

plt.scatter(data_test[:,0], data_test[:,1], color='r')
plt.scatter(data_test[:,0], r.currTar[0], color='g')

# print em.model._mu[:,0]
# print em.model._mu[:,1]
# print em.model._mu[:,2]

# print em.model._sigma[:,:,0]
# print em.model._sigma[:,:,1]
# print em.model._sigma[:,:,2]


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # # Make data.
# # X = np.arange(-5, 5, 0.25)
# # Y = np.arange(-5, 5, 0.25)
# # X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)

# # # Plot the surface.
# # ax.plot_surface(X, Y, Z, 
# #                        linewidth=0, antialiased=False)

# ax.scatter(5, 10, 12, marker='x')

plt.show()