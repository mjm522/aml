import numpy as np
import pybullet as pb
from config import config
from aml_math.quaternion_utils import compute_omg

class BulletRobot(object):

    def __init__(self, robot_id, ee_link_idx=-1, update_rate=100, config=config, enable_force_torque_sensors = False):

        self._config = config

        self._id    = robot_id

        self._state = None

        self.configure_camera()

        self._ee_link_idx = ee_link_idx

        self._joint_idx = self.get_movable_joints()

        self._force_torque_sensors_enabled = [False for n in self._joint_idx]

        if enable_force_torque_sensors:
            self.enable_force_torque_sensors()

        
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


        # print "Projection: \n", np.asarray(self._projection_matrix, dtype=np.float32).reshape(4,4)


    def get_image(self):

        _, _, self._view_matrix, _, _, _, _, _, _, _, _, _ = pb.getDebugVisualizerCamera()
        (w, h, rgb_pixels, depth_pixels, _) = pb.getCameraImage(self._config['cam']['image_width'], 
                                    self._config['cam']['image_height'], 
                                    self._view_matrix, self._projection_matrix, [0,1,0])

        # rgba to rgb
        rgb_image = rgb_pixels[:,:,0:3]
        depth_image  = depth_pixels 


        return rgb_image, depth_image


    # ----- If joint_idx is not specified, sensors for all joints are enabled
    def enable_force_torque_sensors(self, joint_idx = -2):

        if joint_idx == -2:
            joint_idx = self._joint_idx

        if isinstance(joint_idx, int) and joint_idx not in self._joint_idx:
            raise Exception("Invalid Joint ID")

        for joint in joint_idx:
            pb.enableJointForceTorqueSensor(self._id, joint, 1)
            self._force_torque_sensors_enabled[joint] = True

    # ----- If joint_idx is not specified, sensors for all joints are disabled
    def disable_force_torque_sensors(self, joint_idx = -2):

        if joint_idx == -2 or joint_idx == self._joint_idx:
            joint_idx = self._joint_idx

        if isinstance(joint_idx, int) and joint_idx not in self._joint_idx:
            raise Exception("Invalid Joint ID")

        for joint in joint_idx:
            pb.enableJointForceTorqueSensor(self._id, joint, 0)
            self._force_torque_sensors_enabled[joint] = False

    # --- Gives pos, vel, joint force, joint torque, motor torque values of all joints (unless joint_idx is specified). 
    # -------- If flag is set to 'all', the output (for each joint) is of the following format: [pos, vel, (fx, fy, fz, tx, ty, tz), motor_torque]
    def get_joint_details(self, joint_idx = -2, flag = 'all'):

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

    def _update_state(self, event = None):

        class Time:
            def __init__(self):
                self.secs = 0
                self.nsecs = 0

        now                      = Time()#rospy.Time.now()

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


    def get_state(self):
        return self._state

    def get_ee_velocity(self):
        #this is a simple finite difference based velocity computation
        #please note that this might produce a bug since self._goal_ori_old gets 
        #updated only if get_ee_vel is called. 
        #TODO : to update in get_ee_pose or find a better way to compute velocity
        
        time_now_new = self.get_time_in_seconds()
        
        ee_pos_new, ee_ori_new = self.get_ee_pose()  

        dt = time_now_new-self._time_now_old

        ee_vel = (ee_pos_new - self._ee_pos_old)/dt

        ee_omg = compute_omg(ee_ori_new, self._ee_ori_old)/dt

        self._goal_ori_old = ee_ori_new
        self._goal_pos_old = ee_pos_new
        self._time_now_old = time_now_new
        
        return ee_vel, ee_omg

    def get_ee_velocity_from_bullet(self):

        if self._ee_link_idx == -1:

            ee_vel, ee_omg = self.get_base_vel()

        else:

            ee_vel, ee_omg = self.get_link_velocity(self._ee_link_idx)

        return ee_vel, ee_omg 

    def get_base_vel(self):

        return pb.getBaseVelocity(self._id)

    def get_time_in_seconds(self):
        # time_now =  rospy.Time.now()

        return 0.0
        # return time_now.secs + time_now.nsecs*1e-9


    def get_movable_joints(self):

        movable_joints = []
        for i in range (pb.getNumJoints(self._id)):
            jointInfo = pb.getJointInfo(self._id, i)
            qIndex = jointInfo[3]
            if qIndex > -1 and jointInfo[1] != "head_pan":
                movable_joints.append(i)

        return movable_joints

    def get_ee_pose(self):

        if self._ee_link_idx == -1:

            pos, ori = self.get_base_pos_ori()

        else:

            pos, ori = self.get_link_pose(self._ee_link_idx)

        return pos, ori 

    # ----- If link_id is not specified, end effector pose is returned
    def get_link_pose(self, link_id = -3):        

        if link_id == -3:
            self._ee_link_idx

        # if link_id not in self._joint_idx:
        #     raise Exception("Invalid Link ID")

        link_state = pb.getLinkState(self._id, link_id)
        pos = np.asarray(link_state[0]) 
        ori = np.asarray(link_state[1])

        return pos, ori 

    def get_link_velocity(self, link_id = -3):

        if link_id == -3:
            self._ee_link_idx

        # if link_id not in self._joint_idx:
        #     raise Exception("Invalid Link ID")

        link_state = pb.getLinkState(self._id, link_id, computeLinkVelocity = 1)
        # print "this====\n",link_state

        lin_vel = np.asarray(link_state[6]) 
        ang_vel = np.asarray(link_state[7])

        return lin_vel, ang_vel


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

        num_jnts = len(self._joint_idx)

        jnt_pos = []
        jnt_vel = []
        jnt_reaction_forces =  []
        jnt_applied_torque  = []

        for jnt_idx in range(len(self._joint_idx)):

            jnt_state = pb.getJointState(self._id, self._joint_idx[jnt_idx])
            jnt_pos.append(jnt_state[0])
            jnt_vel.append(jnt_state[1])
            jnt_reaction_forces.append(jnt_state[2])
            jnt_applied_torque.append(jnt_state[3])

        return jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque


    def set_jnt_state(self, jnt_state):

        num_jnts = len(self._joint_idx)

        if len(jnt_state) < num_jnts:
            print "Pass %d values", num_jnts
            raise ValueError

        else:
            for jnt_idx in range(num_jnts):
                pb.resetJointState(self._id, jnt_idx, jnt_state[jnt_idx])
                

    def move_using_pos_control(self, joints, des_joint_values):
        vels = [0.0005 for n in range(len(joints))]
        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.POSITION_CONTROL, targetPositions= des_joint_values, targetVelocities = vels )


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

    def set_joint_velocities(self, cmd, joints = -3):

        if joints == -3:
            joints = self._joint_idx

        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.VELOCITY_CONTROL, targetVelocities = cmd)

    def set_joint_torques(self, cmd, joints = -3):

        if joints == -3:
            joints = self._joint_idx

        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.TORQUE_CONTROL, forces = cmd)

    def get_inertia_matrix(self, joints = None):

        if joints == None:

            joints = self._state['position']

        elif len(joints) < len(self._joint_idx):
            print "Pass %d values", len(self._joint_idx)
            raise ValueError

        return pb.calculateMassMatrix(self._id, joints)
