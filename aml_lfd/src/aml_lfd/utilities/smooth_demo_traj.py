import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class SmoothDemoTraj():
    """
    A class to smoothen the demonstrated trajectories
    """

    def __init__(self, traj2smooth, window_len=5, poly_order=2):
        """
        Constructor of the class 
        Args: 
        traj_to_smooth: expects np.array([no_of_data_points, number_of_dimension])
        window_len : length of the convolutional window (has to be an odd number)
        poly_order : any number less than the window len and > 0
        """
        self._traj2smooth = traj2smooth
        self._num_steps, self._num_dof = traj2smooth.shape

        self._smoothed_traj = np.zeros_like(self._traj2smooth)

        self._window_len = window_len
        self._poly_order = poly_order

        self.smooth_traj()


    def smooth_traj(self):
        """
        This function takes each dimension and smooths it out
        """

        for k in range(self._num_dof):
          self._smoothed_traj[:, k] = self.savitsky_gollay_filter(self._traj2smooth[:, k])  


    
    def savitsky_gollay_filter(self, traj):
        """
        this is a smoothing filter that helps to make 
        the exploratory trajectories smooth.
        This is an optional part and is implemented depending on the self._smooth_traj variable
        Args:
        traj: input trajectory
        """
        return savgol_filter(x=traj, window_length=self._window_len, polyorder=self._poly_order)

    
    def plot(self):
        """
        plot the original and smoothe trajectory
        """
        plt.figure("Smooth traj comparison")
        subplot_num = min(9, self._num_dof)*100+11

        if self._num_dof > 9:

            print "A single plot window can only show 9 subplots, so showing only first 9 plots"

        for k in range(min(9, self._num_dof)):
            plt.subplot(subplot_num)
            subplot_num += 1

            plt.plot(self._traj2smooth[:, k], 'r')
            plt.plot(self._smoothed_traj[:, k], 'g')

        plt.show()

