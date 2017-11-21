import numpy as np

from aml_planners.push_planner.costs.cost import Cost


class CostImp(Cost):


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


    def cost(q, u, du, t):
        '''
        computes uncontrolled state cost for a trajectory
        '''

        dist           = (np.linalg.norm(q[:2]-self._obstacle_pos )**2-self._obstacle_radius**2)

        vel_state_cost = self._vel_state_cost_coeff*np.linalg.norm(q[3:])**2

        dist_goal      = np.linalg.norm(q[:3]-goal[:3])**2
        
        goal_cost      = self._goal_cost_coeff * dist_goal

        control_cost   = self._control_cost_coeff * np.dot(u, np.dot(self._R,u))

        obstacle_cost  =  self._obstacle_cost_coeff * np.exp(-dist)

        r              = goal_cost +  obstacle_cost +  control_cost + vel_state_cost

        # print "Cost contribution goal:=%f, obstacle:=%f, control:=%f velocity=%f"%(goal_cost, obstacle_cost, control_cost, vel_state_cost)

        if t == (self._N-1):
            r += self._final_cost_coeff*goal_cost

        if (r>1e9) or np.isnan(r):
            r = 1e9

        return r



