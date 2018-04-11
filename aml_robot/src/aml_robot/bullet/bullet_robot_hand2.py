import sys
import numpy as np
import quaternion
import pybullet as pb
from aml_robot.bullet.bullet_robot2 import BulletRobot2
from utils import parse_state
from glove_interface.config import default_glove_config as g_config

import atexit


def to_bullet_quat(q, flip_z=False):
    return [q.x, q.y, q.z, q.w]
    # if flip_z:
    #     return [q.x, q.y, -q.z, q.w]#np.flip(quaternion.as_float_array(q),0)
    # else:


def from_bullet_to_np_quat(array):
    return quaternion.quaternion(array[3], array[0], array[1], array[2])


def make_np_quat(array):
    return quaternion.quaternion(array[0], array[1], array[2], array[3])


class BulletRobotHand(BulletRobot2):
    def __init__(self, robot_id, config):

        BulletRobot2.__init__(self, robot_id, config)

        self._config = config

        self._ori_offset = quaternion.from_euler_angles(*config['orientation_offset'])

        self._pos = np.array([0.0, 0., 0.])
        self._ori_quat = np.quaternion(1, 0, 0, 0)
        self._start_pos = np.array([0.0, 0., 0.])
        self._start_ori = quaternion.from_euler_angles(0, 1.570796327, 3.141592654)

        self.configure_default_pos(np.array([0.0, 0., 0.]), to_bullet_quat(self._ori_offset * self._ori_quat))

        self._thumb_joints = [self.get_joint_by_name(jm) for jm in self._config['thumb_joints']]

        self._index_joints = [self.get_joint_by_name(jm) for jm in self._config['index_joints']]

        self._middle_joints = [self.get_joint_by_name(jm) for jm in self._config['middle_joints']]

        self._ring_joints = [self.get_joint_by_name(jm) for jm in self._config['ring_joints']]

        self._little_joints = [self.get_joint_by_name(jm) for jm in self._config['little_joints']]

        self._joint_map = {"thumb": self._thumb_joints,
                           "index": self._index_joints,
                           "middle": self._middle_joints,
                           "ring": self._ring_joints,
                           "little": self._little_joints}

        self._all_joints = [self._joint_map[finger_name] for finger_name in self._config["finger_order"]]

        self._nfingers = len(self._config["finger_order"])
        # Hack for the right hand
        # self._all_joints.reverse()

        self._joint_state = np.zeros(len(self._joints))

        self._joint_limits = self.get_joint_limits(self._joints)

        self._finger_forces = np.zeros(len(self._config['force_joint_ids']))

        self._force_ratio = np.zeros(len(self._config['force_joint_ids']))

        self.add_static_debug_visual_elements()

        pb.getCameraImage(640, 480, renderer=pb.ER_BULLET_HARDWARE_OPENGL)

        self._ori_offset_sliders = []
        self._ori_offset_sliders.append(
            pb.addUserDebugParameter("ori_offset_roll", 0, 2 * np.pi, self._config['orientation_offset'][0]))
        self._ori_offset_sliders.append(
            pb.addUserDebugParameter("ori_offset_pitch", 0, 2 * np.pi, self._config['orientation_offset'][1]))
        self._ori_offset_sliders.append(
            pb.addUserDebugParameter("ori_offset_yaw", 0, 2 * np.pi, self._config['orientation_offset'][2]))

        atexit.register(self.on_shutdown)

    def add_static_debug_visual_elements(self):

        for jn in self._joints:
            pb.addUserDebugLine([0, 0, 0], [0, 0.0, 0.05], [1, 0, 0], parentObjectUniqueId=self._hand_id,
                                parentLinkIndex=jn, lineWidth=0.05, lifeTime=0)
            pb.addUserDebugText("jnt_%d" % jn, [0, 0, -0.025], textColorRGB=[1, 0, 0], textSize=1.0,
                                parentObjectUniqueId=self._hand_id, parentLinkIndex=jn)

    def update_debug_visual_elements(self, state):

        for idx, jn in enumerate(self._config['force_joint_ids']):

            bid = g_config['bids'][idx]
            max_v = np.maximum(self._finger_forces[idx] + 1, state[bid] + 1)
            min_v = np.minimum(self._finger_forces[idx] + 1, state[bid] + 1)

            threshold = self._config['force_vis_threshold']

            self._finger_forces[idx] = state[bid]
            self._force_ratio[idx] = 1.0 - (min_v / max_v)

            if self._force_ratio[idx] > threshold:
                pb.addUserDebugLine([0, 0, 0], [0, 0.0, self._config['force_scale'] * self._finger_forces[idx]],
                                    [0, 1, 0], parentObjectUniqueId=self._hand_id, parentLinkIndex=jn, lineWidth=5.0,
                                    lifeTime=0.1)
                # print "force ratio: ", self._force_ratio

    def get_joint_limits(self, joints):

        joint_lims = [self.get_joint_limits(joint_idx) for joint_idx in joints]
        return dict(zip(joints, joint_lims))

    def set_state(self, state):

        euler = [0, 0, 0]
        for i in range(len(self._ori_offset_sliders)):
            euler[i] = pb.readUserDebugParameter(self._ori_offset_sliders[i])
            self._ori_offset = quaternion.from_euler_angles(*euler)

        # print self._all_joints
        # print state[g_config['fids']]
        for i in range(self._nfingers):

            if self._config["finger_order"][i] != "thumb":
                joints = self._all_joints[i][1:]  # do not set abduction joint for other fingers
            else:
                joints = self._all_joints[i][0:]  # thumb abduction joint

            self._joint_state[joints] = self.convert_value(state[g_config['fids'][i]], joints)

        pos = state[g_config['pos_ids']]
        # ori_euler = np.array(state[14:18])


        if self._config['update_pos']:
            self._pos = pos

        if self._config['update_ori']:
            # self._ori_euler = self._ori_euler*0.95 + ori_euler*0.05
            self._ori_quat = from_bullet_to_np_quat(state[g_config['ori_ids']]) * self._ori_offset
        else:
            self._ori_quat = self._ori_offset * self._start_ori.conjugate()

        # quat=make_np_quat([-0.69284895,  0.72001595, -0.004938  ,  0.038897  ])
        # print to_bullet_quat(self._ori_offset*self._ori_quat)

        quat = to_bullet_quat(self._ori_quat)
        self.set_pos_ori(self._pos, quat)  # [ 0.0328642, -0.17682831, 0.05765395, -0.98200232])

        # print self._robot.get_base_pos_ori()


        # msg = self._listener_conn.recv()
        # # do something with msg
        # if msg == 'close':
        #     self._listener_conn.close()
        # else:
        #     if msg[:3] == 'th_':
        #         joints = self._thumb_joints[1:]
        #     elif msg[:3] == 'in_':
        #         joints = self._index_joints[1:]
        #     elif msg[:3] == 'mi_':
        #         joints = self._middle_joints[1:]
        #     elif msg[:3] == 'ri_':
        #         joints = self._ring_joints[1:]
        #     elif msg[:3] == 'li_':
        #         joints = self._little_joints[1:]

        #     self._joint_state[joints] = self.convert_slider_value(float(msg[3:]), joints)

    def setup_gui(self):
        phys_id = pb.connect(pb.SHARED_MEMORY)

        if (phys_id < 0):
            phys_id = pb.connect(pb.UDP,
                                 "127.0.0.1")  # "192.168.1.3", 1234) #"127.0.0.1"#cid = p.connect(p.UDP,"192.168.86.100")

    def convert_value(self, value, joints):

        if self._invert_state_val:
            value = 1.0 - value

        normalized_value = value
        cmd = np.zeros(len(joints))

        cmd = np.asarray([(1.0 - normalized_value) * self._joint_limits[joint]['min'] + normalized_value *
                          self._joint_limits[joint]['max']
                          for joint in joints])

        return cmd


    def on_shutdown(self):

        # this if for saving files in case keyboard interrupt happens
        # self._record_sample.save_data_now()
        pb.removeBody(self._hand_id)
        pb.removeBody(self._world_id)

        for item in self._ori_offset_sliders:
            print "Removing item", item
            pb.removeUserDebugItem(item)

        pb.removeAllUserDebugItems()

        pb.resetSimulation()

        pb.disconnect()

    # def sample_action_finger(self, finger_name='thumb'):

    #     if finger_name == 'thumb':
    #         joints = self._thumb_joints
    #     elif finger_name == 'index':
    #         joints = self._index_joints
    #     elif finger_name == 'middle':
    #         joints = self._middle_joints
    #     elif finger_name == 'ring':
    #         joints = self._ring_joints
    #     elif finger_name == 'little':
    #         joints = self._little_joints
    #     else:
    #         joints = self._joints

    #     return np.random.randn(len(joints)),joints

    def update(self, action=None, joints=None):

        # if (action is None) or (joints is None):
        #     return

        self.set_jnt_state(self._joint_state)
        # self._robot.velocity_ctrlr(action, joints)
    def run(self):
        while True:

            try:
                for line in iter(sys.stdin.readline, ''):
                    # line = sys.stdin.readline()
                    # print "Line received: ", line
                    # print "Reading stuff"
                    state = parse_state(line)
                    # print "State received: ", state
                    self.set_state(state)

                    self.update_debug_visual_elements(state)

                    self.update()

                    sys.stdin.flush()
            except:
                break







