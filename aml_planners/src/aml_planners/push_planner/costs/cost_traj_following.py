import numpy as np

from aml_planners.push_planner.costs.cost import Cost
import pickle


class CostTrajFollowing(Cost):


    def __init__(self, config):


        self._N = config['N']

        self._R = config['R']
        # self._Q = config['Q']
        self._nu = config['nu']
        self._xT = config['goal']
        self._obstacle_pos = config['obstacle']
        self._obstacle_radius = config['obstacle_radius']

        self._goal_cost_coeff     = config['goal_cost_coeff']
        self._vel_state_cost_coeff = config['vel_state_cost_coeff']
        self._final_cost_coeff    = config['final_cost_coeff']
        self._control_cost_coeff  = config['control_cost_coeff']
        self._obstacle_cost_coeff = config['obstacle_cost_coeff']
        self._uncertain_cost_coeff = config['uncertain_cost_coeff']

        self._traj = pickle.load( open( config['traj_path'], "rb" ) )

        self._curr_goal_index = 0

        self._c = 0


    def next_goal(self):

        if self._curr_goal_index < (self._traj.shape[0]-1):
            self._curr_goal_index += 1

        print "Moving forward to next goal!!!!"

    def get_goal(self):

        if self._curr_goal_index < self._traj.shape[0]:
            return self._traj[self._curr_goal_index,:]
        else:
            return self._traj[-1,:]

    def get(self, q, u, du, sigma, t):
        '''
        computes uncontrolled state cost for a trajectory
        '''

        

        goal = self._traj[self._curr_goal_index,:]

        # print "Traje info: ", self._traj.shape, goal

        vel_state_cost = self._vel_state_cost_coeff*np.linalg.norm(q[3:])**2

        dist_goal      = np.linalg.norm(q[:2]-goal)**2
        
        goal_cost      = self._goal_cost_coeff * dist_goal

        # control_cost   = self._control_cost_coeff * np.dot(u, np.dot(self._R,u))

        du_cost = (1.0 - 1.0/self._nu)*np.dot(du, np.dot(self._R,du)) + np.dot(u, np.dot(self._R,du))


        r              = goal_cost  + vel_state_cost #+ du_cost

        # print "Cost contribution goal:=%f, obstacle:=%f, control:=%f velocity:=%f uncertainity_cost:=%f"%\
                                # (goal_cost, obstacle_cost, control_cost, vel_state_cost, uncertainity_cost)


        if t == (self._N-1):
            r += self._final_cost_coeff*goal_cost

        if (r>1e9) or np.isnan(r):
            r = 1e9


        # print "Current distance to goal: ", np.sqrt(dist_goal)
        # if np.sqrt(dist_goal) <= 0.2:
        #     self.next_goal()


        return r



