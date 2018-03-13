import tf
import rospy
import numpy as np
import pybullet as pb
from sensor_msgs.msg import JointState 
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent, OmniFeedback, OmniState


class PhantomOmni(object):
    """
    Interface class to the Phantom Omni device
    """

    def __init__(self, ori_aml_format=False, scale=1e-2):

        """
        Constructor of the class
        Args: ori_aml_format : if the return should be aml_format or 
        ros_format
        """

        self._omni_state = {'ee_pos': None, 'ee_vel': None, 'ee_ori': None}

        self._omni_js_state = {'names':[], 'js_pos':np.zeros(6)}

        self._omni_bt_state = {'grey_bt':None, 'white_bt':None}

        self._ori_aml_format = ori_aml_format

        #these values are from the baxter urdf file
        self._jnt_limits = [{'lower':-0.98,  'upper':0.98},
                            {'lower':0.,     'upper':1.75},
                            {'lower':-0.81,  'upper':1.25},
                            {'lower':3.92,   'upper':8.83},
                            {'lower':-0.5,   'upper':1.75},
                            {'lower':-2.58,  'upper':2.58}]

        self._jnt_home = [0., 0.26888931, -0.63970184, 6.28073207, 1.56147123, 0.55215159]

        self._pos = np.zeros(3)

        self._tf_listener = tf.TransformListener()

        self._omni_bt_sub    = rospy.Subscriber("/phantom/button", OmniButtonEvent, self.omni_bt_callback)
        
        self._omni_js_sub    = rospy.Subscriber("/phantom/joint_states", JointState, self.omni_js_callback)

        self._omni_pos_sub   = rospy.Subscriber("/phantom/pose", PoseStamped, self.omni_pos_callback)

        self._omni_state_sub = rospy.Subscriber("/phantom/state", OmniState, self.omni_state_callback)

        self._omni_ffbk_pub  = rospy.Publisher("/phantom/force_feedback", OmniFeedback, queue_size=10)

        self._scale = scale

        #this flag is to for mapping into a calibration
        #space
        self._calibrated = False

        #while doing the demonstrations, the white button
        #has to be pressed
        self._device_enabled = False

        self._calib_pos = (0.,0.,0.)

        self._calib_ori = (0., 0., 0., 1.)

        self._update_state() 


    def omni_bt_callback(self, msg):

        self._omni_bt_state['grey_bt']  = msg.grey_button  and 1
        
        self._omni_bt_state['white_bt'] = msg.white_button and 1

        if self._omni_bt_state['white_bt']:

            self._device_enabled = True

        else:

            self._device_enabled = False


    def omni_js_callback(self, msg):
        
        time_stamp =  msg.header.stamp

        for k in range(6):
            
            self._omni_js_state['names'].append(msg.name[k])
            
            self._omni_js_state['js_pos'][k] = msg.position[k]


    def _update_state(self):

        print "HERE"

        now                = rospy.Time.now()

        state = {}
        
        state['position']  = self.get_jnt_state()

        state['velocity']  = None

        state['effort']    = None

        state['applied']   = None

        # TODO : interface with omni_pykdl to get the jacobian
        state['jacobian']  = None 

        state['timestamp'] = { 'secs' : now.secs, 'nsecs': now.nsecs }

        state['ee_point'], state['ee_ori']  = self.get_ee_pose()

        state['ee_vel']  = self._omni_state['ee_vel']
        
        state['ee_omg']  = None

        self._state = state


    def get_tf_transform(self, frame1='/base', frame2='/stylus'):

        pos = None; ori = None

        if self._tf_listener.frameExists(frame1) and self._tf_listener.frameExists(frame1):

            try:

                time = self._tf_listener.getLatestCommonTime(frame1, frame2)
                
                pos, ori = self._tf_listener.lookupTransform(frame1, frame2, time)
            
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                
                pass
        
        else:

            print "The given frames does not exist!"

        return pos, ori


    def calibration(self, ee_pos, ee_ori):
        """
        This function is for arbitary transformations
        between arbitarary ee_pos and ee_ori
        to the ee_pos and ee_ori of phantom omni

        Args: ee_pos = tuple(x,y,z)
              ee_ori = tuple(x,y,z,w)
        """
        if isinstance(ee_pos, np.ndarray):
            ee_pos = tuple(ee_pos)

        if isinstance(ee_ori, np.ndarray):
            
            ee_ori = tuple(ee_ori)

        elif isinstance(ee_ori, np.quaternion):

            ee_ori = (ee_ori.x, ee_ori.y, ee_ori.z, ee_ori.w)

        omni_ee_pos, omni_ee_ori = self.get_ee_pose()

        inv_hap_pos, inv_hap_ori = pb.invertTransform(tuple(omni_ee_pos),
                                                      tuple(omni_ee_ori))

        self._calib_pos, self._calib_ori = pb.multiplyTransforms(inv_hap_pos, inv_hap_ori, ee_pos, ee_ori)

        print "************************* Calibrated transformations **************************"
        print "Pos \t", self._calib_pos
        print "Ori \t", self._calib_ori
        print "*******************************************************************************"

        self._calibrated = True

    def get_ee_pose_calib_space(self):

        """
        Once calibrated, we could transform the points
        arbitarily, the current ee_pos and ee_ori of the
        haptic device will be mapped into the calibrated
        space
        """

        if not self._calibrated:

            print "The calibration is not performed"

        omni_ee_pos, omni_ee_ori = self.get_ee_pose()

        pos, ori = pb.multiplyTransforms(tuple(omni_ee_pos), 
                                        tuple(omni_ee_ori), 
                                        self._calib_pos, 
                                        self._calib_ori)

        if self._ori_aml_format:

            ori = (ori[3], ori[0], ori[1], ori[2])

        return np.asarray(pos), np.asarray(ori)


    def get_jnt_state(self):

        if self._omni_js_state['js_pos'] is None:

            js_pos = np.zeros(6)

        else:

            js_pos = self._omni_js_state['js_pos']

        return js_pos

    def get_ee_pose(self):

        if self._omni_state['ee_pos'] is not None:

            ee_pos = self._scale*self._omni_state['ee_pos']

        else:

            ee_pos = None

        return ee_pos, self._omni_state['ee_ori']


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

        #if the white button is not pressed
        #the states wont be updated and hence 
        #the haptic device maintains the last data
        if self._device_enabled:

            self._update_state()


    def omni_force_feedback(self, force, gain=0.5):

        force_msg = OmniFeedback()
        
        force_msg.force.x = min(max(force[0]*gain, -3.0), 3.0)
        
        force_msg.force.y = min(max(force[1]*gain, -3.0), 3.0)
        
        force_msg.force.z = min(max(force[2]*gain, -3.0), 3.0)

        self._omni_ffbk_pub.publish(force_msg)
 