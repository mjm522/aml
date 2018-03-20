import os
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

MOUSE_MOVE_EVENT = 1
MOUSE_BUTTON_EVENT = 2
KEY_IS_DOWN = 3
KEY_WAS_RELEASED = 4


def get_demo(demo_path):
    """
    load the demo trajectory from the file
    """

    if not os.path.isfile(demo_path):
        raise Exception("The given path to demo does not exist, given path: \n" + demo_path)

    demo_data  = np.genfromtxt(demo_path, delimiter=',')

    return demo_data

def plot_demo(trajectory, color=[0,0,1], start_idx=8, life_time=0.):
    """
    this funciton is to load trajectory into the bullet viewer.
    the state is a list of 6 values, only the x,y,z values are taken
    Args: 
    trajectory = np.array([no of point, no of dimension])
    start_idx = start location of the 3d position array
    life_time = life time of the line in seconds
    """

    for k in range(trajectory.shape[0]-1):

        # print "Data point:", k, ":", trajectory[k, start_idx:start_idx+3]

        draw_trajectory(point_1=trajectory[k, start_idx:start_idx+3], 
                        point_2=trajectory[k+1, start_idx:start_idx+3], 
                        colour=color, line_width=5.5, life_time=life_time)


def draw_trajectory(point_1, point_2, colour=[0,1,0], line_width=4.5, life_time=0.):
    """
    This function adds colour line between points point_1 and point_2 in the bullet
    Args:
    point_1: starting point => [x,y,z]
    point_2: ending_point => [x,y,z]
    """
    pb.addUserDebugLine(point_1, point_2, lifeTime=life_time, lineColorRGB=colour, lineWidth=line_width)


class CollectDemo():

    def __init__(self, manipulator, demo_path=None):

        if demo_path is None:
            self._demo_folder_path = './demo'
        else:
            self._demo_folder_path = demo_path

        if not os.path.exists(self._demo_folder_path):
            os.makedirs(self._demo_folder_path)

        #demo collecting variables
        self._left_button_down = False
        self._demo_collection_start = False
        self._demo_point_count = 0

        self._manipulator = manipulator
        
        #variable required for correct operation
        self._traj_point_1, _  = self._manipulator.ee_pose()

        #for saving data
        self._ee_pos_array = []
        self._ee_vel_array = []
        self._js_pos_array = []
        self._js_vel_array = []


    def collect_demo(self, demo_draw_interwal=10):
        """
        This function is for collecting the demo trajectory. It works with the help of mouse events.
        To record a demo, click the left button on the mouse and drag the end effector of the manipulator.
        To stop and save the recorded demo, simply release the left button press.
        Args: 
        demo_draw_interwal = this parameter decides at what interwal the plot needs to be updated
        """
        #get the tuple of mouse events
        mouse_events = pb.getMouseEvents()

        #check the tuple only if its length is more than zero
        
        if len(mouse_events) > 0:

            if mouse_events[0][0] == MOUSE_BUTTON_EVENT:
                
                #left button
                if mouse_events[0][3] == 0:

                    #button is down
                    if mouse_events[0][4] == KEY_IS_DOWN:

                        keyboard_events = pb.getKeyboardEvents()

                        #if left control key is pressed it is not for demo collecting
                        #it is to rotate the screen
                        if 65307L in keyboard_events.keys():

                            return    

                        else:

                            self._left_button_down = True

                    #button is released
                    if mouse_events[0][4] == KEY_WAS_RELEASED:

                        self._left_button_down = False

            #collect the demo only if the button was pressed
            if self._left_button_down:

                if mouse_events[0][0] == MOUSE_MOVE_EVENT:

                    self._demo_collection_start = True

                    print "Start collecting demo"

                    #start collecting demo point
                    self._demo_point_count += 1

                    traj_point_2, _ = self._manipulator.ee_pose()

                    ee_vel, ee_omg = self._manipulator.ee_velocity()

                    jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()

                    print "traj_point", traj_point_2

                    self._ee_pos_array.append(traj_point_2)
                    self._ee_vel_array.append(ee_vel)
                    self._js_pos_array.append(jnt_pos)
                    self._js_vel_array.append(jnt_vel)
 
                    #draw the lines in specific interwal
                    if self._demo_point_count % demo_draw_interwal == 0:

                        draw_trajectory(point_1=self._traj_point_1, point_2=traj_point_2)

                        #store the previous value to draw from this point
                        self._traj_point_1 = traj_point_2
            
            else:

                #stop the collection only if it was started
                if self._demo_collection_start:

                    print "Stop collecting demo"

                    data = np.hstack([np.round(np.asarray(self._js_pos_array).squeeze() ,3), 
                                      np.round(np.asarray(self._js_vel_array).squeeze() ,3), 
                                      np.round(np.asarray(self._ee_pos_array).squeeze(), 3), 
                                      np.round(np.asarray(self._ee_vel_array).squeeze(), 3),])

                    #the data is smoothed using savitsky_gollay_filter
                    smooth_demo_data = SmoothDemoTraj(data)

                    file_name = self._demo_folder_path + '/block_js_ee_pos_data'+'.csv'

                    #save the smoothed data
                    np.savetxt(file_name, smooth_demo_data._smoothed_traj, delimiter=",")

                    self._demo_collection_start = False
                    self._demo_point_count = 0

                    self._ee_pos_array = []
                    self._ee_vel_array = []

                    return True
                    
        return False



class SmoothDemoTraj():
    """
    A class to smoothen the demonstrated trajectories
    """

    def __init__(self, traj2smooth):
        """
        Constructor of the class 
        Args: 
        traj_to_smooth: expects np.array([no_of_data_points, number_of_dimension])
        """
        self._traj2smooth = traj2smooth
        self._num_steps, self._num_dof = traj2smooth.shape

        self._smoothed_traj = np.zeros_like(self._traj2smooth)

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
        return savgol_filter(x=traj, window_length=5, polyorder=2)

    
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

