import tf
import rospy
import numpy as np
from sensor_msgs.msg import JointState 
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent, OmniFeedback, OmniState


class PhantomOmni(object):

    def __init__(self, ori_aml_format=False, update_rate=500):

        rospy.init_node('omni_state_listener', anonymous=True)

        self._omni_state = {'ee_pos': None, 'ee_vel': None, 'ee_ori': None}

        self._omni_js_state = {'names':[], 'js_pos':np.zeros(6)}

        self._omni_bt_state = {'grey_bt':None, 'white_bt':None}

        self._ori_aml_format = ori_aml_format

        self._omni_start_ee_pos = None
        self._omni_start_ee_ori = None
        self._omni_start_ee_vel = None

        self._rel_ee_pos = np.zeros(3)
        self._rel_ee_vel = np.zeros(3)
        self._rel_ee_ori = np.zeros(4)

        #these values are from the baxter urdf file
        self._jnt_limits = [{'lower':-0.98,  'upper':0.98},
                            {'lower':0.,     'upper':1.75},
                            {'lower':-0.81,  'upper':1.25},
                            {'lower':3.92,   'upper':8.83},
                            {'lower':-0.5,   'upper':1.75},
                            {'lower':-2.58,  'upper':2.58}]

        self._state = None

        _update_period = rospy.Duration(1.0/update_rate)

        rospy.Timer(_update_period, self._update_state)

        self._pos = np.zeros(3)

        self._tf_listener = tf.TransformListener()

        self._omni_bt_sub = rospy.Subscriber("/phantom/button", OmniButtonEvent, self.omni_bt_callback)
        
        self._omni_js_sub = rospy.Subscriber("/phantom/joint_states", JointState, self.omni_js_callback)

        self._omni_pos_sub = rospy.Subscriber("/phantom/pose", PoseStamped, self.omni_pos_callback)

        self._omni_state_sub = rospy.Subscriber("/phantom/state", OmniState, self.omni_state_callback)

        self._omni_ffbk_pub = rospy.Publisher("/phantom/force_feedback", OmniFeedback, queue_size=10)


    def omni_bt_callback(self, msg):

        self._omni_bt_state['grey_bt']  = msg.grey_button  and 1
        
        self._omni_bt_state['white_bt'] = msg.white_button and 1


    def omni_js_callback(self, msg):
        
        time_stamp =  msg.header.stamp

        for k in range(6):
            
            self._omni_js_state['names'].append(msg.name[k])
            
            self._omni_js_state['js_pos'][k] = msg.position[k]


    def _update_state(self, event):

        now                = rospy.Time.now()

        state = {}
        
        state['position']  = self.get_jnt_state()

        state['velocity']  = None

        state['effort']    = None

        state['applied']   = None

        state['jacobian']  = None

        state['timestamp'] = { 'secs' : now.secs, 'nsecs': now.nsecs }

        state['ee_point'], state['ee_ori']  = self.get_ee_pose()

        state['ee_vel']  = self._omni_state = ['ee_vel']
        
        state['ee_omg']  = None

        self._state = state


    def get_tf_transform(self, frame1='/base', frame2='/stylus'):

        pos = None
        ori = None

        if self._tf_listener.frameExists(frame1) and self._tf_listener.frameExists(frame1):

            try:

                time = self._tf_listener.getLatestCommonTime(frame1, frame2)
                
                pos, ori = self._tf_listener.lookupTransform(frame1, frame2, time)
            
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                
                pass
        
        else:

            print "The given frames does not exist!"

        return pos, ori


    def get_tf_frame(self, frame='/base'):

        pos = None
        ori = None

        if self._tf_listener.frameExists(frame):

            pos, ori = self._tf_listener.getFrames(frame)

        return pos, ori


    def get_jnt_state(self):

        if self._omni_js_state['js_pos'] is None:

            js_pos = np.zeros(6)

        else:

            js_pos = self._omni_js_state['js_pos']

        return js_pos

    
    def get_ee_pose(self):

        return self._omni_state = ['ee_pos'], self._omni_state = ['ee_ori']


    def omni_pos_callback(self, msg):
        
        self._pos = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])


    def omni_state_callback(self, msg):
        
        time_stamp = msg.header.stamp

        lock_status = msg.locked

        gipper_close_status = msg.close_gripper
        
        pos = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        if self._ori_aml_format:
            
            ori = np.asarray([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        
        else:
            
            ori = np.asarray([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

        vel = np.asarray([msg.velocity.x, msg.velocity.y, msg.velocity.z])

        #the default unit is in mm
        self._omni_state['ee_pos'] = pos

        self._omni_state['ee_vel'] = vel

        self._omni_state['ee_ori'] = ori


    def omni_force_feedback(self, force, gain=0.5):

        force_msg = OmniFeedback()
        
        force_msg.force.x = min(max(force[0]*gain, -3.0), 3.0)
        force_msg.force.y = min(max(force[1]*gain, -3.0), 3.0)
        force_msg.force.z = min(max(force[2]*gain, -3.0), 3.0)

        self._omni_ffbk_pub.publish(force_msg)

    def update_omni_state(self, start=False):

        if start:
            
            self._omni_start_ee_pos = self._omni_state['ee_pos']
            self._omni_start_ee_ori = self._omni_state['ee_ori']
            self._omni_start_ee_vel = self._omni_state['ee_vel']

            self._omni_curr_ee_pos  = np.zeros(3)
            self._omni_curr_ee_ori  = np.zeros(4)
            self._omni_curr_ee_vel  = np.zeros(3)

        else:

            self._omni_curr_ee_pos = self._omni_state['ee_pos']
            self._omni_curr_ee_ori = self._omni_state['ee_ori']
            self._omni_curr_ee_vel = self._omni_state['ee_vel']

            self._rel_ee_pos = self._omni_curr_ee_pos - self._omni_start_ee_pos
            self._rel_ee_ori = self._omni_curr_ee_ori - self._omni_start_ee_ori
            self._rel_ee_vel = self._omni_curr_ee_vel - self._omni_start_ee_vel

            self._omni_start_ee_pos = self._omni_curr_ee_pos
            self._omni_start_ee_ori = self._omni_curr_ee_ori
            self._omni_start_ee_vel = self._omni_curr_ee_vel 


