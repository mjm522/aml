import os
import numpy as np
import pandas as pd
import pylab as plt
from forward_models.simple_nn_model import SimpleNNModel

# fix random seed for reproducibility
np.random.seed(42)

df = pd.DataFrame.from_csv(os.environ['MPPI_DATA_DIR']+'/trans_push_data_all_rand.csv')
X = np.array(df[['xi', 'yi']])
actions = np.array(df[['action']])
X = np.hstack([X, actions])
Y = np.array(df[['xf', 'yf']])

#pre_processing stage
X_train = X.copy()
Y_train = np.zeros([Y.shape[0], 3])

for k in range(Y.shape[0]):
    delta = Y[k,:] - X[k,0:2]
    Y_train[k,2]  = np.linalg.norm(delta) 
    Y_train[k,0:2] = delta/np.linalg.norm(delta)



network_params = {
    'load_saved_model':True,
    'model_path':os.environ['MPPI_DATA_DIR']+'/keras_models/simple_nn_trans_push_data_all_rand.h5',
    'epochs':500,
    'batch_size':20,
    'save_model':True,
    'state_dim':2,
    'cmd_dim':1,
    'out_dim':3,
}


config = {'network_params': network_params,
          'random_seed':0,}

simple_nn = SimpleNNModel(config)
loss = simple_nn.fit(X=X_train, y=Y_train)

predictions, sigmas = simple_nn.predict(X_train)

if not network_params['load_saved_model']:
    plt.figure(1)
    plt.plot(loss)

simple_nn.plot(Y_train, predictions)