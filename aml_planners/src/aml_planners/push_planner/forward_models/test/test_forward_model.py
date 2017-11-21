import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from forward_models.gp_model import GPModel
from forward_models.dropoutNN_model import DropoutNN
from forward_models.ensemble_model import EnsambleModel

ForwardModel = EnsambleModel

if ForwardModel == EnsambleModel:
    from config import ensemble_config as config
elif ForwardModel == DropoutNN:
    from config import dropout_config as config
elif ForwardModel == GPModel:
    config = 2


file_name = os.environ['MPPI_DATA_DIR']+'box2d_push_data_all_rand_circle.csv'


#get data
df = pd.DataFrame.from_csv(file_name)
X = np.array(df[['xi', 'yi', 'thetai']])
actions = np.array(df[['action']])
X = np.hstack([X, actions])
Y = np.array(df[['xf', 'yf', 'thetaf']])


#pre_processing stage
X_train = X.copy()
Y_train = Y - X[:,0:3]

fwd_model = ForwardModel(config)

print "Fitting model"
# Fitting model
fwd_model.fit(X_train, Y_train)

### test on training data
# train_mus, train_sigmas = fwd_model.predict(X)
# plt.figure("Training data test")
# plt.subplot(311)
# plt.plot(Y_train[:,0], c='r')
# plt.plot(train_mus[:,0], c='g')
# plt.subplot(312)
# plt.plot(Y_train[:,1], c='r')
# plt.plot(train_mus[:,1], c='g')
# plt.subplot(313)
# plt.plot(Y_train[:,2], c='r')
# plt.plot(train_mus[:,2], c='g')

# plt.show()
# plt.scatter(np.concatenate([y_test[:,0],y[:,0]],axis=0), np.concatenate([y_test[:,1],y[:,1]],axis=0), s=80, c='g', marker='+')

print "Generating heat map"
## Heat map
pxs = np.linspace(0., 7., 100)
pys = np.linspace(0., 7., 100)
X, Y = np.meshgrid(pxs, pys)
positions = np.vstack([X.ravel(), Y.ravel()]).T
positions = np.hstack([positions, np.random.uniform(0., 2*np.pi, positions.shape[0])[:,None]])

XS_Test = np.hstack([positions, np.random.uniform(0, 1, positions.shape[0])[:,None]])

mus, sigmas = fwd_model.predict(XS_Test)

noise_level = 0.001
for i in range(4):

    x = positions + np.random.randn(positions.shape[0],3)*noise_level
    u = np.random.uniform(0, 1, x.shape[0])[:,None]

    XS_Test = np.hstack([x, u])

    mus_tmp, sigmas_tmp = fwd_model.predict(XS_Test)

    sigmas += sigmas_tmp

sigmas /= 5.0

if ForwardModel == EnsambleModel:
    sigmas = np.sum(sigmas,axis=1)
elif ForwardModel == GPModel:
    sigmas = np.sum(sigmas,axis=0)

# plt.figure(1)
# plt.contour(positions[:,0],positions[:,1],np.reshape(sigmas,(-1,1)))

sigmas = np.reshape(sigmas,(100,100))

plt.figure("Heatmap")
plt.imshow(sigmas, origin='lower', interpolation='nearest', extent=[0.,7.,0.,7.])
plt.colorbar()

plt.show()