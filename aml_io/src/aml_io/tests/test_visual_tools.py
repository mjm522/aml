import numpy as np
from aml_io.visual_tools import *


#test for plotting 2D data
data = np.random.randn(2,10).tolist()
fig_handle = visualize_2D_data(data, fig_handle=None, axis_lim=None)

#test for plotting 3D data
data = np.random.randn(3, 100).tolist()
fig_handle = visualize_3D_data(data, fig_handle=None, axis_lim=None)

#test for plotting continous 2D data


#test for plotting continous 3D data


#test for showing image