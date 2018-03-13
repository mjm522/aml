import numpy as np
import pybullet as pb
from scipy.optimize import minimize
from utils import unit_normal, poly_area
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG


class QPPosGaits():

    def __init__(self):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False,  keep_obj_fixed=True)
        
        else:
            
            self._env = env

        HAND_OBJ_CONFIG['renders'] = False

        self._dummy_env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False,  keep_obj_fixed=True)
        
        hand_info = self._env.get_hand_limits()

        self._num_fingers = self._env._num_fingers

        self._finger_joint_means  = hand_info['mean']
        
        self._finger_joint_ranges = hand_info['range']
        
    def get_hand_manipulability(self, finger_idx, new_finger_joint_pos):
        
        qh = 0.

        curr_joint_pos = self._env.get_hand_joint_state()['pos']

        curr_joint_pos[finger_idx] = new_finger_joint_pos

        for k in range(self._num_fingers):

            for j in range(self._num_joints_finger):

                qh += ((curr_joint_pos[k][j] - self._finger_joint_means[k,j])/self._finger_joint_ranges[k,j])**2


        return -0.5*qh


    def get_contact_points(self):

        contact_info = self._env.get_contact_points()

        contact_points = np.ones([self._num_fingers, 3])*np.nan

        for k in range(self._num_fingers):

            if contact_info[k]['cp_on_finger']:

                contact_points[k, :] = np.round(np.asarray(contact_info[k]['cp_on_finger'][0]), 6)

        return contact_points


    def get_grasp_quality(self, finger_idx, new_finger_ee_pos):
        """
        finger index should be the free index
        this assumes all four fingers are in contact
        """

        contact_points = self.get_contact_points()

        contact_points[finger_idx,:] = new_finger_ee_pos

        area = poly_area(contact_points)

        if np.isnan(area):
            area = 0.

        return area


    def run(self, finger_idx, weight=0.2):

        contact_point = self.get_contact_points()[finger_idx, :]

        # print "The contact point \t", contact_point

        closest_points = self._env.get_closest_points()

        lower_limit   = self._finger_limits['lower'][finger_idx][:3]
        upper_limit   = self._finger_limits['upper'][finger_idx][:3]

        # print "Current j pos:\t", self._env.get_ik(finger_idx, contact_point)

        cost_function = lambda x: -(1.-weight)*self.get_hand_manipulability(finger_idx, self._env.get_ik(finger_idx, np.asarray(x)) ) - weight* self.get_grasp_quality(finger_idx, np.asarray(x) ) 

        bounds = ((contact_point[0]-0.01, contact_point[0]+0.01), (contact_point[1]-0.01, contact_point[1]+0.01), (contact_point[2]-0.00001, contact_point[2]+0.00001))

        constraints = ({'type': 'ineq', 'fun': lambda x: lower_limit - self._env.get_ik(finger_idx, np.asarray(x))},
                       {'type': 'ineq', 'fun': lambda x: self._env.get_ik(finger_idx, np.asarray(x)) - upper_limit})

        res = minimize(cost_function, tuple(contact_point), method='SLSQP', bounds=bounds, constraints=constraints)

        return self._env.get_ik(finger_idx, res['x'])



def main():
    qpg = QPPosGaits()

    """
    this does only with the continous joints specified in the urdf
    """

    while True:

        for k in range(qpg._num_fingers):

            new_js_pos = qpg.run(k)

            if np.any(np.isnan(new_js_pos)):
                print "Skipped"
                continue

            print "New j pos:",k,":\t", new_js_pos

            qpg._env._hand.applyAction(k, new_js_pos)

            p.stepSimulation()

if __name__ == '__main__':
    main()