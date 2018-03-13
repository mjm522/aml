import numpy as np
import pybullet as pb
from scipy.optimize import minimize, linprog
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG

np.random.seed(123)

class QPVelGaits():

    def __init__(self, env=None):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False,  keep_obj_fixed=True)
        
        else:
            
            self._env = env

        hand_info = self._env.get_hand_limits()

        self._num_fingers = self._env._num_fingers

        self._finger_joint_means  = hand_info['mean']
        
        self._finger_joint_ranges = hand_info['range']
    
    def get_contact_points(self):

        contact_info = self._env.get_contact_points()

        contact_points = np.ones([self._num_fingers, 3])*np.nan

        for k in range(self._num_fingers):

            if contact_info[k]['cp_on_finger']:

                contact_points[k, :] = np.round(np.asarray(contact_info[k]['cp_on_finger'][0]), 6)

        return contact_points

    
    def get_Qh_dot(self, finger_idx, new_js_vel):
        
        #equation 7

        qh_dot = 0.

        curr_joint_pos = self._env.get_hand_joint_state()['pos'][finger_idx]

        for j in range(self._num_joints_finger-1):

            mean_j = self._finger_joint_means[finger_idx,j]
            min_j  = self._finger_limits['lower'][finger_idx][j]
            max_j  = self._finger_limits['upper'][finger_idx][j]

            q_thresh=0.25*(max_j-min_j)

            if (curr_joint_pos[j] - mean_j) < -q_thresh:
                c_j = np.log( (mean_j-min_j-q_thresh)/(curr_joint_pos[j] - min_j) ) + 1.
            elif np.abs(curr_joint_pos[j] - mean_j) <= q_thresh:
                c_j = 1.
            else:
                c_j = np.log( (max_j-mean_j-q_thresh)/(max_j-curr_joint_pos[j])  ) + 1.

            #equation 7, part 2
            qh_dot += c_j*( ( mean_j - curr_joint_pos[j])/(max_j-min_j)**2)*new_js_vel[j]

        return qh_dot


    def get_Qo_dot(self, finger_idx, new_js_vel):
        """
        finger index should be the free index
        this assumes all four fingers are in contact
        """
        contact_points = self.get_contact_points()

        #choose other three points except the one that is being changed
        p1, p2, p3 =  [x for i,x in enumerate(contact_points) if i != finger_idx]

        pb = self._env.get_hand_ee_state()['pos'][finger_idx]
        #equation 6 # normal n1 is perpendicular to p2p3 and in plane p1p2p3, joining line p2p3 and pIb
        theta = np.arccos( (np.dot( (p3-p2),(p3-pb) ))/( np.linalg.norm(p2-p3) * np.linalg.norm(p3-pb)) )

        normal_len = np.sin(theta) * np.linalg.norm(pb-p3)

        #finding the ratio by which the normal would divide line p3->2
        alpha = np.dot((pb-p3),(p2-p3))/np.linalg.norm(p2-p3)

        #finding a point that divides the line p3->p2 by the ratio alpha, which gives one end of the normal (other end is pb)
        normal_vector = pb - (p3 + alpha*(p2-p3)) 

        #equation 7, part 1
        qo_dot = np.dot((np.linalg.norm(p2-p3)*normal_vector).T, self.compute_ee_vel_from_jac(finger_idx, new_js_vel))

        return qo_dot


    def get_jacobian(self, finger_idx, local_point=None):
        
        curr_joint_pos = self._env.get_hand_joint_state()['pos']

        return self._env.get_finger_jac(finger_idx=finger_idx, j_poss=curr_joint_pos, local_point=local_point)


    def compute_ee_vel_from_jac(self, finger_idx, new_js_vel):
        
        ee_state_jac   = self.get_jacobian(finger_idx)

        return np.dot(ee_state_jac, new_js_vel)


    def run(self, finger_idx, weight=0.1):

        print "Finger idx \t", finger_idx

        curr_js_vel   = np.asarray(self._env.get_hand_joint_state()['vel'][finger_idx][:3])
        curr_ee_vel   = self._env.get_hand_ee_state()['vel'][finger_idx]

        # J = self.get_jacobian(finger_idx=finger_idx, new_js_vel=np.zeros_like(curr_js_vel))
        # print "Curr ee vel \t", curr_ee_vel
        # print "Jac ee vel \t", np.dot(J, curr_js_vel)

        #in rad/sec
        lower_bound = -1.
        upper_bound = 1.

        a = np.vstack([np.ones(3)*lower_bound, np.ones(3)*upper_bound]).T
        bounds = tuple([tuple(a[k]) for k in range(3)])

        def constraint_2(new_js_vel):

            #jacobian
            J = self.get_jacobian(finger_idx=finger_idx)
        
            contact_info = self._env.get_contact_points()[finger_idx]

            if contact_info['cn_on_block']:
                
                contact_force_dir_block = np.asarray(contact_info['cn_on_block'][0])
                
                contact_point = np.asarray(contact_info['cp_on_block'][0])
                
                obj_pos, obj_ori, _, _, _, _ = self._env.get_obj_curr_state(ori_as_euler=False)

                contact_force_dir  = np.dot( np.linalg.inv(obj_ori), (contact_force_dir_block-obj_pos) ) + contact_point
            
            else:
                contact_force_dir = np.zeros(3)

            return np.dot(np.dot(contact_force_dir, J), new_js_vel)


        #the optimizer available is minimize, hence we maximize the cost function
        cost_function = lambda x: -( (1.-weight)*self.get_Qh_dot(finger_idx, x) + weight* self.get_Qo_dot(finger_idx, x) ) 

        constraints = ({'type': 'ineq', 'fun': lambda x: np.linalg.norm(curr_ee_vel - self.compute_ee_vel_from_jac(finger_idx, x), ord=np.inf) - 0.002 },
                       {'type': 'eq',   'fun': lambda x: constraint_2(x)})

        #tuple(curr_js_vel) tuple(np.random.randn(3))
        #SLSQP
        #TNC = kinds of move, but not right
        res = minimize(cost_function, tuple(curr_js_vel), method='SLSQP', bounds=bounds, constraints=constraints)

        # res = linprog(cost_function, tuple(curr_js_vel), method='interior-point', bounds=bounds, constraints=constraints)

        print "*********************Status Qp vel gaits**************************************", res['success']

        if res['success']:
            return res['x'], curr_js_vel
        else:
            return curr_js_vel, curr_js_vel



def main():

    qpg = QPVelGaits()

    while True:

        for k in range(qpg._num_fingers):

            new_js_vel, _ = qpg.run(k)

            print "New j vel:",k,":\t", new_js_vel

            if np.any(np.isnan(new_js_vel)):
                print "Skipped"
                continue

            qpg._env._fingers[k].applyAction(new_js_vel)

            p.stepSimulation()

if __name__ == '__main__':
    main()