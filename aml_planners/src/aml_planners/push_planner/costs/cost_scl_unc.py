import numpy as np

from aml_planners.push_planner.costs.cost import Cost


class CostScaledUnc(Cost):


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

        self._use_cutoff_heuristic = config['use_cutoff_heuristic']
        self._cutoff_heuristic_thr = config['cutoff_heuristic_thr']
        self._c = 0

        print config


    def get(self, q, u, du, sigma, t):
        '''
        computes uncontrolled state cost for a trajectory
        '''



        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        dist           = (np.linalg.norm(q[:2]-self._obstacle_pos)**2-self._obstacle_radius**2)

        vel_state_cost = self._vel_state_cost_coeff*np.linalg.norm(q[3:])**2

        dist_goal      = np.linalg.norm(q[:3]-self._xT[:3])**2
        
        goal_cost      = self._goal_cost_coeff * dist_goal

        control_cost   = self._control_cost_coeff * np.dot(u, np.dot(self._R,u))

        du_cost = (1.0 - 1.0/self._nu)*np.dot(du, np.dot(self._R,du)) + np.dot(u, np.dot(self._R,du))

        obstacle_cost  =  self._obstacle_cost_coeff * np.exp(-dist)


        uncertainity_cost = self._uncertain_cost_coeff * (sigma**2)#np.exp(np.square(sigma - 1.0))#np.sum(np.sqrt(sigma))

        # np.exp(1.0/((sigma-6.3)*self._uncertain_cost_coeff))
        r              = min(goal_cost*(1.0/sigmoid(((sigma-12.8)*self._uncertain_cost_coeff))),100.0) +  obstacle_cost +  control_cost + vel_state_cost + 300.0*sigmoid(((sigma-12.8)*self._uncertain_cost_coeff))#+ uncertainity_cost #+ du_cost

        # print "Cost contribution goal:=%f, obstacle:=%f, control:=%f velocity:=%f uncertainity_cost:=%f"%\
                                # (goal_cost, obstacle_cost, control_cost, vel_state_cost, uncertainity_cost)



        if t == (self._N-1):
            r += self._final_cost_coeff*goal_cost

        if (r>1000.0) or np.isnan(r):
            r = 1000.0

        return r



