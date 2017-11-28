import cv2
import rospy
import numpy as np
import pybullet as pb
from config import config

class BulletRobot(object):

    def __init__(self, robot_id, ee_link_idx=-1, update_rate=100, config=config):

        self._config = config

        self._id    = robot_id

        self._state = None

        self.configure_camera()

        self._ee_link_idx = ee_link_idx

        _update_period = rospy.Duration(1.0/update_rate)

        rospy.Timer(_update_period, self._update_state)

        
    def configure_default_pos(self, pos, ori):

        self._default_pos = pos
        self._default_ori = ori
        self.set_default_pos_ori()


    def configure_camera(self):

        self._view_matrix = pb.computeViewMatrixFromYawPitchRoll(self._config['cam']['target_pos'], 
                                                                 self._config['cam']['distance'], 
                                                                 self._config['cam']['yaw'], 
                                                                 self._config['cam']['pitch'], 
                                                                 self._config['cam']['roll'], 
                                                                 self._config['cam']['up_axis_index'])
        
        _aspect = self._config['cam']['image_width'] / self._config['cam']['image_height']

        self._projection_matrix = pb.computeProjectionMatrixFOV(self._config['cam']['fov'], 
                                                                _aspect, 
                                                                self._config['cam']['near_plane'], 
                                                                self._config['cam']['far_plane'])


    def get_image(self, display_image=False):

        img_arr = pb.getCameraImage(self._config['cam']['image_width'], 
                                    self._config['cam']['image_height'], 
                                    self._view_matrix, self._projection_matrix, [0,1,0])


        rgba_image   = np.asarray(img_arr[2], dtype=np.uint8).reshape(img_arr[0],img_arr[1], 4)
        depth_image  = np.asarray(img_arr[3], dtype=np.float32).reshape(img_arr[0], img_arr[1])

        if display_image:
            cv2.imshow("captured image", rgba_image)
            cv2.waitKey()

        # image = {'width': img_arr[0],
        #          'height':img_arr[1],
        #          'rgba':np.asarray(img_arr[2],  dtype=np.uint8),
        #          'depth':np.asarray(img_arr[3], dtype=np.float32)}

        return rgba_image, depth_image


    def _update_state(self, event):

        state = {}

        now                      = rospy.Time.now()

        state = {}
        
        state['position'], state['velocity'],\
        state['effort'],   state['applied'] = self.get_jnt_state()

        state['jacobian']        = None
        state['inertia']         = None
        # state['rgb_image'], state['depth_image']  = self.get_image()
        state['gravity_comp']    = None
        state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }
        state['ee_point'], state['ee_ori']  = self.get_ee_pose()

        state['ee_vel']  = None
        state['ee_omg']  = None


        self._state = state

    def get_ee_pose(self):

        if self._ee_link_idx == -1:

            pos, ori = self.get_pos_ori()

        else:

            link_state = pb.getLinkState(self._id, self._ee_link_idx)
            pos = np.asarray(link_state[0]) 
            ori = np.asarray(link_state[1])

        return pos, ori 

    def get_pos_ori(self):

        pos, ori = pb.getBasePositionAndOrientation(self._id)

        return np.asarray(pos), np.asarray(ori)

    def get_contact_points(self):

        contact_info = list(pb.getContactPoints(self._id))

        if contact_info:

            contact_info = list(contact_info[0])

        return contact_info

    def set_default_pos_ori(self):

        self.set_pos_ori(self._default_pos, self._default_ori)

    def set_pos_ori(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._id, pos, ori)


    def get_jnt_state(self):

        num_jnts = pb.getNumJoints(self._id)

        jnt_pos = np.zeros(num_jnts)
        jnt_vel = np.zeros(num_jnts)
        jnt_reaction_forces =  np.zeros(num_jnts)
        jnt_applied_torque  = np.zeros(num_jnts)

        for jnt_idx in range(num_jnts):

            jnt_state = pb.getJointState(self._id, jnt_idx)

            jnt_pos[jnt_idx] = jnt_state[0]
            jnt_vel[jnt_idx] = jnt_state[1]
            jnt_reaction_forces[jnt_idx]=jnt_state[2]
            jnt_applied_torque[jnt_idx] = jnt_state[3]

        return jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque


    def set_jnt_state(self, jnt_state):

        num_jnts = pb.getNumJoints(self._id)

        if len(jnt_state) < num_jnts:
            print "Pass %d values", num_jnts
            raise ValueError

        else:
            for jnt_idx in range(num_jnts):

                pb.resetJointState(self._id, jnt_idx, jnt_state[jnt_idx])

    def apply_external_force(self, link_idx, force, point, local=True):

        if local:

            pb.applyExternalForce(self._id, link_idx, force, point , pb.LINK_FRAME)

        else:

            pb.applyExternalForce(self._id, link_idx, force, point , pb.WORLD_FRAME)


    def apply_external_torque(self, link_idx, torque, point, local=True):

        if local:

            pb.applyExternalTorque(self._id, link_idx, force, point , pb.LINK_FRAME)

        else:

            pb.applyExternalTorque(self._id, link_idx, force, point , pb.WORLD_FRAME)
