import os
import GPy
import random
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(101)

#get data
df = pd.DataFrame.from_csv(os.environ['MPPI_DATA_DIR']+'trans_2_push_data_all_rand_gap.csv')
X = np.array(df[['xi', 'yi']])
actions = np.array(df[['action']])
X = np.hstack([X, actions])
Y = np.array(df[['xf', 'yf']])


# df_test = pd.DataFrame.from_csv(os.environ['MPPI_DATA_DIR']+'trans_2_push_data_all_rand.csv')
# X = np.array(df_test[['xi', 'yi']])
# actions_test = np.array(df_test[['action']])
# X = np.hstack([X, actions_test])
# Y = np.array(df_test[['xf', 'yf']])

pxs = np.linspace(-1, 1., 100)
pys = np.linspace(-1, 1., 100)
X_t, Y_t = np.meshgrid(pxs, pys)
positions = np.vstack([X_t.ravel(), Y_t.ravel()]).T
X_test = np.hstack([positions, np.random.uniform(0,1, positions.shape[0])[:,None]])

X_scaled = X#preprocessing.scale(X)
Y_scaled = Y#preprocessing.scale(Y)


# kern = GPy.kern.RBF(1, variance=np.random.exponential(1.), lengthscale=np.random.exponential(50.))
# kern = GPy.kern.RBF(1, variance=np.random.uniform(0.1, 0.9), lengthscale=np.random.uniform(1, 50))


#ful gaussain fit
m_full_dx = GPy.models.GPRegression(X=X,  Y=Y[:,0][:,None], noise_var=.1)
m_full_dy = GPy.models.GPRegression(X=X,  Y=Y[:,1][:,None], noise_var=.1)
# m_full_mag = GPy.models.GPRegression(X, Y[:,2][:,None])

print "Log likelihood of m_full_dx \t", m_full_dx.log_likelihood()
print "Log likelihood of m_full_dy \t", m_full_dy.log_likelihood()

m_full_dx.optimize('bfgs')
m_full_dy.optimize('bfgs')
# m_full_mag.optimize('bfgs')

X_test_scaled =  X_test#preprocessing.scale(X_test)
# X_test_scaled = np.vstack([X_test_scaled, X_scaled])

mus_dx,  sigmas_dx = m_full_dx.predict(X_test_scaled)
mus_dy,  sigmas_dy = m_full_dy.predict(X_test_scaled)
# mus_mag, sigmas_mag = m_full_mag.predict(X)


# fig = plt.figure("GPy test")
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(X_scaled[:,0], X_scaled[:,1], c='b')
# plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c='g')
# plt.scatter(Y_test[:,0], Y_test[:,1], c='r')
# plt.scatter(mus_dx[:,0], mus_dy[:,0], c='g')


# for k in range(mus_dx.shape[0]):
#     circ = plt.Circle((X_test_scaled[k,0], X_test_scaled[k,1]), radius=(sigmas_dx[k,0]+sigmas_dy[k,0]), color='grey', alpha=0.08)
#     ax.add_patch(circ)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:,0], X_scaled[:,1], np.zeros(X_scaled.shape[0]), c='r', marker='*')
ax.scatter(X_test_scaled[:,0], X_test_scaled[:,1], sigmas_dx+sigmas_dy, c='c', marker='o')

# Plot the surface.
# fig2 = plt.figure("variance")
# ax2 = fig2.gca(projection='3d')
# surf = ax2.plot_surface(X_test_scaled[:,0], X_test_scaled[:,1], sigmas_dx+sigmas_dy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig2.colorbar(surf, shrink=0.5, aspect=5)


plt.show()


print adadfa

##poor sparse fit
Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
m = GPy.models.SparseGPRegression(X,y,Z=Z)
m.likelihood.variance = noise_var
m.plot()
print m


#optimizing covariance params
m.inducing_inputs.fix()
m.optimize('bfgs')
m.plot()
print m


#optimizing inducing inputs
m.randomize()
m.Z.unconstrain()
m.optimize('bfgs')
m.plot()


#train with more data points
Z = np.random.rand(12,1)*12
m = GPy.models.SparseGPRegression(X,y,Z=Z)

m.optimize('bfgs')
m.plot()
m_full.plot()
print m.log_likelihood(), m_full.log_likelihood()

plt.show()