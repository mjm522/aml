import rospy
import numpy as np
from aml_services.srv import PredictAction, PredictState
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files

def predict_state_client(state, action):
    rospy.wait_for_service('predict_state')
    try:
        predict_state_service = rospy.ServiceProxy('predict_state', PredictState)
        response = predict_state_service(state, action)
        return response.next_state
    except rospy.ServiceException, e:
        print "Service call to predict_state failed: %s"%e


def predict_action_client(curr_state, tgt_state):
    rospy.wait_for_service('predict_action')
    try:
        predict_action_service = rospy.ServiceProxy('predict_action', PredictAction)
        response = predict_action_service(curr_state, tgt_state)
        return response.action
    except rospy.ServiceException, e:
        print "Service call to predict_action failed: %s"%e


def main():

    x_curr   = np.random.randn(1,7).tolist()[0]
    x_tgt    = np.random.randn(1,7).tolist()[0]
    u_action = np.random.randn(1,2).tolist()[0]

    x_nxt = predict_state_client(x_curr, u_action)
    u_act = predict_action_client(x_curr, x_tgt)

    print "Next state predicted by state predictor service is \t", x_nxt
    print "Action predicted by action predictor service is \t", u_act


if __name__ == '__main__':
    main()