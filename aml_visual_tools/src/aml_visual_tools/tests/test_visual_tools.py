import numpy as np
from aml_visual_tools.visual_tools import *


#test for plotting 2D data
data_xy = np.random.randn(2,10)
fig_handle = visualize_2D_data(data=data_xy, fig_handle=None, axis_lim=None)

#test for plotting 3D data
data_xyz = np.random.randn(3, 100)
fig_handle = visualize_3D_data(data=data_xyz, fig_handle=None, axis_lim=None)

#test for plotting 2D data with variance
sigma = np.random.randn(1,10)
fig_handle = visualize_2D_data_with_sigma(data=data_xy, sigma=sigma, fig_handle=None, axis_lim=None)

#test for plotting continous 2D data


#test for plotting continous 3D data


#test for showing image