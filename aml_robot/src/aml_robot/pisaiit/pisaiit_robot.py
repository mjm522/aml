# General ROS imports
import roslib
roslib.load_manifest('aml_robot')
import rospy
from std_msgs.msg import Float32

# AML additional imports
from aml_robot.robot_interface import RobotInterface
from aml_io.log_utils import aml_logging



class PisaIITHand(RobotInterface):
    def __init__(self, robot_name, on_state_callback=None):
        """
        Class constructor
        Args: 
        robot_name: a string (ideally unique and human readable) representing this robot name
        on_state_callback: an optional callback
        Returns:
        none, store the trajectories
        """

        self._logger = aml_logging.get_logger(__name__)

        self._ready = False

        # Configuring hand (setting up publishers, variables, etc)
        self._configure(robot_name, on_state_callback)

        self._ready = True  # Hand is ready to be used

    def _update_state(self):

        # now                 = rospy.Time.now()

        # joint_angles        = self.angles()
        # joint_velocities    = self.joint_velocities()
        # joint_efforts       = self.joint_efforts()

        # joint_names         = self.joint_names()

        # def to_list(ls):
        #     return [ls[n] for n in joint_names]

        state = {}
        # state['position']        = joint_angles
        # state['velocity']        = np.array(to_list(joint_velocities))
        # state['effort']          = np.array(to_list(joint_efforts))
        # state['jacobian']        = self.get_jacobian_from_joints(None)
        # state['inertia']         = self.get_inertia(None)
        # state['rgb_image']       = self._camera._curr_rgb_image
        # state['depth_image']     = self._camera._curr_depth_image
        # state['gravity_comp']    = np.array(self._h)


        # state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        # try:
        #     state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        # except:
        #     pass

        # try:
        #     state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
        # except:
        #     pass

        return state

    def _configure(self, limb, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None


        # self._kinematics = baxter_kinematics(self)


        self._pos_cmd_pub = rospy.Publisher('soft_hand_pos_cmd', Float32, queue_size=10)
        self._sh_current_status = rospy.Publisher('soft_hand_read_current', Float32, queue_size=10)


        # self._camera = camera_sensor.CameraSensor()

        # self._ee_force = None
        # self._ee_torque = None

    def _on_joint_states(self, msg):

        # Updates internal state

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)

    def exec_position_cmd(self, cmd):
        self._pos_cmd_pub(float(cmd))

    def exec_position_cmd_delta(self, cmd):
        self._logger.warning("Position command delta not implemented.")

    def exec_velocity_cmd(self, cmd):
        self._logger.warning("Velocity commands not implemented.")

    def exec_torque_cmd(self, cmd):
        self._logger.warning("Torque commands not implemented.")

    def angles(self):
        pass

    def q_mean(self):
        pass

    def inertia(self, joint_angles=None):
        pass

    def cartesian_velocity(self, joint_velocities=None):
        pass

    def forward_kinematics(self, joint_angles=None):
        pass

    def inverse_kinematics(self, position, orientation=None):
        pass

    def move_to_joint_pos_delta(self, cmd):
        pass

    def n_cmd(self):
        pass

    def n_joints(self):
        pass

    def tuck(self):
        pass

    def joint_efforts(self):
        pass

    def ee_velocity(self, numerical=False):
        pass

    def untuck(self):
        pass

    def joint_velocities(self):
        pass

    def joint_names(self):
        pass

    def move_to_joint_position(self, cmd):
        pass

    def jacobian(self, joint_angles=None):
        pass

    def state(self):
        return self._state

    def ee_pose(self):
        self._logger.warning("ee_pose commands not implemented.")

    def set_sampling_rate(self, sampling_rate=100):
        pass
