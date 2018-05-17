import os
import rospy
import numpy as np
import pybullet as pb
from aml_io.log_utils import aml_logging
from aml_io.io_tools import get_file_path, get_aml_package_path
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

class RobotBulletInterface():

    def __init__(self, robot_interface, bullet_interface, rate=900):

        self._real_robot = robot_interface

        self._bullet_robot = bullet_interface

        self._master_jlims = self._real_robot._jnt_limits

        self._slave_jlims  = self._master_jlims

        self._num_master_jnts = len(self._master_jlims)

        self._num_slave_jnts = len(self._slave_jlims)

        self._rate = rate

        self._scale_from_home = False

        self._calibrated = False

        self._logger = aml_logging.get_logger(__name__)


    def master_js_scale(self):
        """
        this function finds the js scale of the master robot
        in a range 0 to 1
        """

        scale = np.zeros(self._num_master_jnts)

        curr_js = self._real_robot.state()['position']

        if self._scale_from_home:

            for k in range(self._num_master_jnts):

                scale[k] = (curr_js[k] - self._jnt_home[k])\
                /(self._master_jlims[k]['upper'] - self._master_jlims[k]['lower'])

            scale[scale > 0.5]  = 0.5

            scale[scale < -0.5] = -0.5

        else:

            for k in range(self._num_master_jnts):

                scale[k] = (curr_js[k] - self._master_jlims[k]['lower'])\
                         /(self._master_jlims[k]['upper'] - self._master_jlims[k]['lower'])

            scale[scale > 1.] = 1.

            scale[scale < 0.] = 0.

        return scale

    def compute_cmd(self):
        """
        once the scale is computed it is translated into
        the command of the slave robot
        """

        cmd = np.zeros(self._num_slave_jnts)

        scale = self.master_js_scale()

        if self._scale_from_home:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] += scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        else:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] = self._slave_jlims[j]['lower'] + scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        self._old_cmd = cmd

        return cmd, np.zeros(self._num_slave_jnts)


    def calibration(self, num_samples=1):

        # models_path = get_aml_package_path('aml_lfd/data')
  
        # kwargs = {'path_to_demo':get_file_path('sawyer_bullet_interface_calib.pkl', models_path),
        #           'limb_name':self._real_robot._limb}

        # gen_traj  = JSTrajGenerator(load_from_demo=True, **kwargs)

        # calib_traj = gen_traj.generate_traj()['pos_traj']

        # num_samples = calib_traj.shape[0]

        real_robot_ee_pos = np.zeros([num_samples, 3])
        #in euler
        real_robot_ee_ori = np.zeros([num_samples, 3])

        bullet_robot_ee_pos = np.zeros([num_samples, 3])
        #in euler
        bullet_robot_ee_ori = np.zeros([num_samples, 3])

        for i in range(num_samples):

            # self._real_robot.move_to_joint_position(calib_traj[i,:])

            # self.mirror_real_robot()

            real_robot_state   = self._real_robot.state()

            bullet_robot_state = self._bullet_robot.state()

            real_robot_ori = real_robot_state['ee_ori']
            bullet_robot_ori = bullet_robot_state['ee_ori']
            
            real_robot_ori   = np.asarray(pb.getEulerFromQuaternion((real_robot_ori.x, real_robot_ori.y, real_robot_ori.z, real_robot_ori.w)))
            bullet_robot_ori = np.asarray(pb.getEulerFromQuaternion((bullet_robot_ori.x, bullet_robot_ori.y, bullet_robot_ori.z, bullet_robot_ori.w)))

            real_robot_ee_pos[i,:] = real_robot_state['ee_point']

            real_robot_ee_ori[i,:] = real_robot_ori

            bullet_robot_ee_pos[i,:] = bullet_robot_state['ee_point']
            
            bullet_robot_ee_ori[i,:] = bullet_robot_ori

        calib_pos = np.mean(real_robot_ee_pos-bullet_robot_ee_pos, axis=0)

        calib_ori = np.mean(real_robot_ee_ori-bullet_robot_ee_ori, axis=0)

        base_ori = pb.getQuaternionFromEuler(calib_ori)

        self._logger.info("\nCalib pos \t {}".format(calib_pos))
        self._logger.info("\nCalib ori \t {}".format(base_ori))

        pb.resetSimulation(physicsClientId=self._bullet_robot._cid)

        self._bullet_robot.reset(base_pos=calib_pos.tolist(), base_ori=base_ori)

        self._calibrated = True

    def mirror_real_robot(self):

        goal_js_pos, goal_js_vel = self.compute_cmd()

        self._bullet_robot.apply_jnt_ctrl(cmd=goal_js_pos)

        self._bullet_robot.simple_step()


    def run(self):
        """
        ctrlr function,
        computes the scale and find the required velocity
        """

        rate = rospy.timer.Rate(self._rate)

        if not self._calibrated:

            self.calibration()

        while not rospy.is_shutdown():

            self._bullet_robot.exec_position_cmd(goal_js_pos)#(cmd=goal_js_pos)

            # self._bullet_robot.simple_step()
            
            rate.sleep()

            self.mirror_real_robot()

            rate.sleep()
