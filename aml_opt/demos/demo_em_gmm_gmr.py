import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_opt.em_gmm_gmr.em_gmm_gmr import EM_GMM_GMR

np.random.seed(1337)

means = [ [0, 0], [4, 4], [8, 8] ]
covs =  [ [[0.5, 0.5], [0.5, 0.5]],  [[0.5, 0.5], [0.5, 0.5]],  [[0.5, 0.5], [0.5, 0.5]] ] # diagonal covariance
colors = ['r', 'g', 'b']
data = None
data_test = None

for mean, cov in zip(means, covs):
    
    x, y = np.random.multivariate_normal(mean, cov, 500).T

    if data is None:
        data = np.vstack([x, y]).T
    else:
        data = np.vstack([data, np.vstack([x, y]).T])

for mean, cov in zip(means, covs):
    
    x, y = np.random.multivariate_normal(mean, cov, 200).T

    if data_test is None:
        data_test = np.vstack([x, y]).T
    else:
        data_test = np.vstack([data_test, np.vstack([x, y]).T])

em = EM_GMM_GMR(data=data.T)
em.run_em()
mu_out, sigma_out = em.gmr(data_in=data_test[:,0])

for i in range(3):
    print "Real Guassian %d mu \n"%i, means[i]
    print "Estimated Guassian %d mu \n"%i, em.model._mu[:,i]
    print "*********"
    print "Real Gaussian %d sig \n"%i, covs[i]
    print "Estimated Guassian %d sig \n"%i, em.model._sigma[:,:,i]


plt.scatter(data_test[:,0], data_test[:,1], color='r')
plt.scatter(data_test[:,0], mu_out, color='g', marker='x')

plt.show()