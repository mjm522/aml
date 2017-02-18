import abc
import copy
import rospy
import numpy as np

#this is an abstract class for generating trajectories.
#it interafaces both operation space traj generator and
#joint space traj generator.

class TrajGenerator(object):
    
    __metaclass__ = abc.ABCMeta

    def __init__(self, load_from_demo=False, **kwargs):

        self._traj = {}

        self._time_steps = 0

        self._load_from_demo = load_from_demo

        if load_from_demo:
            #configure required demo index
            self.configure_demo(**kwargs)
        else:
            #configure requried ee points and other spec
            self.configure_traj(**kwargs)

    def configure_demo(self, **kwargs):

        if kwargs.has_key('demo_idx'):

            self._demo_idx  = kwargs['demo_idx']
            self._limb_name = kwargs['limb_name']

        elif kwargs.has_key('path_to_demo'):

            self._demo_path = kwargs['path_to_demo']
            self._limb_name = kwargs['limb_name']

        else:

            print "Asked to load demo, but no demo index found, please pass it"
            raise ValueError


    def configure_traj(self, **kwargs):

        if kwargs.has_key('start_pos') and kwargs.has_key('start_ori') and kwargs.has_key('goal_pos') and kwargs.has_key('goal_ori'):
            
            self._goal_pos  = kwargs['goal_pos']
            self._goal_ori  = kwargs['goal_ori']
            self._start_pos = kwargs['start_pos']
            self._start_ori = kwargs['start_ori']

        else:

            print "Missing either of goal_pos, goal_ori, start_pos or start_ori, please pass them"
            raise ValueError


    @abc.abstractmethod
    def generate_traj(self):
        raise NotImplementedError("Must be implemented in the subclass")