import cv2
import rospy
import numpy as np
import pybullet as pb
from config import config

class BulletRobot(object):

    def __init__(self, robot_id, ee_link_idx=-1, update_rate=100, config=config, enable_force_torque_sensors = False):

        self._config = config

        self._id    = robot_id

        self._state = None

        self.configure_camera()

        self._ee_link_idx = ee_link_idx

        self._joint_idx = [n for n in range(pb.getNumJoints(self._id))]

        self._force_torque_sensors_enabled = [False for n in self._joint_idx]

        if enable_force_torque_sensors:
            self.enableForceTorqueSensors()

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


    # ----- If joint_idx is not specified, sensors for all joints are enabled
    def enableForceTorqueSensors(self, joint_idx = -2):

        if joint_idx == -2:
            joint_idx = self._joint_idx

        if isinstance(joint_idx, int) and joint_idx not in self._joint_idx:
            raise Exception("Invalid Joint ID")

        for joint in joint_idx:
            pb.enableJointForceTorqueSensor(self._id, joint, 1)
            self._force_torque_sensors_enabled[joint] = True

    # ----- If joint_idx is not specified, sensors for all joints are disabled
    def disableForceTorqueSensors(self, joint_idx = -2):

        if joint_idx == -2 or joint_idx == self._joint_idx:
            joint_idx = self._joint_idx

        if isinstance(joint_idx, int) and joint_idx not in self._joint_idx:
            raise Exception("Invalid Joint ID")

        for joint in joint_idx:
            pb.enableJointForceTorqueSensor(self._id, joint, 0)
            self._force_torque_sensors_enabled[joint] = False

    # --- Gives pos, vel, joint force, joint torque, motor torque values of all joints (unless joint_idx is specified). 
    # -------- If flag is set to 'all', the output (for each joint) is of the following format: [pos, vel, (fx, fy, fz, tx, ty, tz), motor_torque]
    def getJointDetails(self, joint_idx = -2, flag = 'all'):

        if joint_idx == -2:
            joint_idx = self._joint_idx

        if isinstance(joint_idx, int):

            if joint_idx not in self._joint_idx:
                raise Exception("Invalid Joint ID")

            details = pb.getJointState(self._id, joint_idx)
            force_torque_sensor_status = self._force_torque_sensors_enabled[joint_idx]
            joint_pos = details[0]
            joint_vel = details[1]
            force_vals = details[2][:3]
            torque_vals = details[2][3:]
            motor_torque = details[3]

        else:
            details = pb.getJointStates(self._id, joint_idx)
            force_torque_sensor_status = []
            joint_pos = []
            joint_vel = []
            force_vals = []
            torque_vals = []
            motor_torque = []
            for joint in joint_idx:
                force_torque_sensor_status.append(self._force_torque_sensors_enabled[joint])
                joint_pos.append(details[joint][0])
                joint_vel.append(details[joint][1])
                force_vals.append(details[joint][2][:3])
                torque_vals.append(details[joint][2][3:])
                motor_torque.append(details[joint][3])

        if flag == 'force_torque_sensor_status':
            return force_torque_sensor_status

        elif flag == 'joint_pos':
            return joint_pos
        elif flag == 'joint_vel':
            return joint_vel

        elif flag == 'force':
            return force_vals

        elif flag == 'torque':
            return torque_vals

        elif flag == 'motor_torque_applied':
            return motor_torque

        elif flag == 'force_torque':
            return force_vals, torque_vals

        elif flag == 'all':
            return details

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

            pos, ori = self.get_base_pos_ori()

        else:

            pos, ori = self.get_link_pose(self._id, self._ee_link_idx)

        return pos, ori 

    # ----- If link_id is not specified, end effector pose is returned
    def get_link_pose(self, link_id = -3):

        if link_id not in self._joint_idx:
            raise Exception("Invalid Link ID")        

        if link_id == -3:
            self._ee_link_idx

        link_state = pb.getLinkState(self._id, link_id)
        pos = np.asarray(link_state[0]) 
        ori = np.asarray(link_state[1])

        return pos, ori 


    def get_base_pos_ori(self):

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




