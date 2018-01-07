import numpy as np

from aml_planners.push_planner.costs.cost import Cost


class CostImp(Cost):


    def __init__(self, config):

        self._N = config['N']

        self._R = config['R']
        # self._Q = config['Q']
        self._nu = config['nu']
        self._xT = config['goal']
        self._x0 = config['start']

        self._obtacle_present = False
        
        if config['obstacle'] is not None:
            self._obtacle_present = True
            self._obstacle_pos = config['obstacle'][:2]

            if len(config['obstacle']) > 2:
                self._obstacle_radius = config['obstacle'][2]
            else:
                self._obstacle_radius = config['obstacle_radius']
            self._obstacle_cost_coeff = config['obstacle_cost_coeff']

        self._goal_cost_coeff     = config['goal_cost_coeff']
        self._goal_ori_coeff      = config['goal_ori_coeff']
        self._vel_state_cost_coeff = config['vel_state_cost_coeff']
        self._final_cost_coeff    = config['final_cost_coeff']
        self._control_cost_coeff  = config['control_cost_coeff']
        
        self._uncertain_cost_coeff = config['uncertain_cost_coeff']

        self._use_cutoff_heuristic = config['use_cutoff_heuristic']
        self._cutoff_heuristic_thr = config['cutoff_heuristic_thr']
        self._c = 0

        print config


    def get(self, q, u, du, sigma, t):
        '''
        computes uncontrolled state cost for a trajectory
        '''

        # target_dir = (self._xT-self._x0)/np.linalg.norm(self._xT-self._x0)
        # curr_dir = (q[:2]-self._x0)/np.linalg.norm(q[:2]-self._xT)
        # dist_goal  = np.linalg.norm(curr_dir-target_dir)**2

        if self._obtacle_present:
            dist           = (np.linalg.norm(q[:2]-self._obstacle_pos)**2-self._obstacle_radius**2)
            obstacle_cost  =  self._obstacle_cost_coeff * np.exp(-dist)
        else:
            obstacle_cost  = 0.

        vel_state_cost = self._vel_state_cost_coeff*np.linalg.norm(q[3:])**2
        
        dist_goal      = np.linalg.norm(q[:2]-self._xT[:2])**2
        # dist_ori       = (np.fmod(q[2],2*np.pi)-np.fmod(self._xT[2],2*np.pi))**2
        # another option: convert to vector and use difference as error
        dist_ori       = np.linalg.norm(np.fmod(q[2],2*np.pi)-np.fmod(self._xT[2],2*np.pi))**2

        goal_cost      = self._goal_cost_coeff * dist_goal + self._goal_ori_coeff*dist_ori

        control_cost   = self._control_cost_coeff * np.dot(u, np.dot(self._R,u))

        du_cost = (1.0 - 1.0/self._nu)*np.dot(du, np.dot(self._R,du)) + np.dot(u, np.dot(self._R,du)) 

        if self._use_cutoff_heuristic and self._c >= self._cutoff_heuristic_thr:
            self._uncertain_cost_coeff = 0


        uncertainity_cost = self._uncertain_cost_coeff * (sigma**2)#np.exp(np.square(sigma - 1.0))#np.sum(np.sqrt(sigma))


        r              = goal_cost +  obstacle_cost +  control_cost + vel_state_cost + uncertainity_cost #+ du_cost

        # print "Cost contribution goal:=%f, obstacle:=%f, control:=%f velocity:=%f uncertainity_cost:=%f"%\
                                # (goal_cost, obstacle_cost, control_cost, vel_state_cost, uncertainity_cost)


        if t == (self._N-1):
            r += self._final_cost_coeff*goal_cost

        if (r>1e9) or np.isnan(r):
            r = 1e9
        if r < 1e-4:
            r = 0.

        return r



