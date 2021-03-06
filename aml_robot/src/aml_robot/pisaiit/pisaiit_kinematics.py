#!/usr/bin/python


import numpy as np
import PyKDL

import rospy

import baxter_interface

from aml_robot.utilities.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF




class pisaiit_kinematics(object):
    """
    Baxter Kinematics with PyKDL
    """

    def __init__(self, limb, description = None):
        if description is None:
            self._pisaiit = URDF.from_parameter_server(key='robot_description')
        else:
            self._pisaiit = URDF.from_xml_file(description)

        self._kdl_tree = kdl_tree_from_urdf_model(self._pisaiit)

        # Robot interface
        self._limb_interface = limb
        self._joint_names = self._limb_interface.joint_names()
        self._num_jnts = len(self._joint_names)


        self._links = self._limb_interface.links()
        self._base_link = self._links['base_link']

        self._finger_links = self._links['finger_links']
        self._tip_links = [self._links[finger_link][-1] for finger_link in self._finger_links]
        self._all_links = []
        for finger_link in self._finger_links:
            self._all_links += self._links[finger_link]

        # self._n_links = [len(self._links[finger_link]) for finger_link in self._finger_links]

        self._tip_frame = PyKDL.Frame()

        self._chains = [self._kdl_tree.getChain(self._base_link, tip_link) for tip_link in self._tip_links]
        self._num_chains = len(self._chains)
        self._chain_num_joints = [chain.getNrOfJoints() for chain in self._chains]
        self._n_links = [chain.getNrOfSegments() for chain in self._chains]
        # KDL Solvers
        self._fk_p_kdl = [PyKDL.ChainFkSolverPos_recursive(chain) for chain in self._chains]
        self._fk_v_kdl = [PyKDL.ChainFkSolverVel_recursive(chain) for chain in self._chains]
        self._ik_v_kdl = [PyKDL.ChainIkSolverVel_pinv(chain) for chain in self._chains]

        self._ik_p_kdl = []
        for i in range(self._num_chains):
            self._ik_p_kdl.append(PyKDL.ChainIkSolverPos_NR(self._chains[i],self._fk_p_kdl[i],self._ik_v_kdl[i]))

        self._jac_kdl = [PyKDL.ChainJntToJacSolver(chain) for chain in self._chains]
        self._dyn_kdl = [PyKDL.ChainDynParam(chain,PyKDL.Vector.Zero()) for chain in self._chains]

    def print_robot_description(self):
        nf_joints = 0
        for j in self._pisaiit.joints:
            if j.type != 'fixed':
                nf_joints += 1
        print "URDF non-fixed joints: %d;" % nf_joints
        print "URDF total joints: %d" % len(self._pisaiit.joints)
        print "URDF links: %d" % len(self._pisaiit.links)
        print "KDL joints: %d" % self._kdl_tree.getNrOfJoints()
        print "KDL segments: %d" % self._kdl_tree.getNrOfSegments()
        print "KDL chains: %d" % self._num_chains

    def print_kdl_chain(self, chain_idx):
        chain = self._chains[chain_idx]
        print "KDL Chain: %d"% chain_idx
        print "KDL joints: %d" % chain.getNrOfJoints()
        print "KDL segments: %d" % chain.getNrOfSegments()
        for idx in xrange(chain.getNrOfSegments()):
            print '* ' + chain.getSegment(idx).getName()

    def joints_to_kdl(self, type, values=None, finger_idx = 0):
        kdl_array = PyKDL.JntArray(self._chain_num_joints[finger_idx])

        if values is None:
            if type == 'positions':
                cur_type_values = self._limb_interface.angles()
            elif type == 'velocities':
                cur_type_values = self._limb_interface.joint_velocities()
            elif type == 'torques':
                cur_type_values = self._limb_interface.joint_efforts()
        else:
            cur_type_values = values

        for idx, name in enumerate(cur_type_values):
            kdl_array[idx] = cur_type_values[idx]
        if type == 'velocities':
            kdl_array = PyKDL.JntArrayVel(kdl_array)
        return kdl_array

    def kdl_to_mat(self, data):
        mat = np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i, j] = data[i, j]
        return mat

    def forward_position_kinematics(self, joint_values=None, finger_idx=0):

        link_poses = []
        for link_idx in range(self.n_links(finger_idx)):
            end_frame = PyKDL.Frame()
            self._fk_p_kdl[finger_idx].JntToCart(self.joints_to_kdl('positions', joint_values, finger_idx),
                                     end_frame, link_idx)
            pos = end_frame.p
            rot = PyKDL.Rotation(end_frame.M)
            rot = rot.GetQuaternion()

            link_poses.append(np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]]))
        return link_poses

    def forward_velocity_kinematics(self, joint_velocities=None, finger_idx=0, link_idx=0):
        end_frame = PyKDL.FrameVel()
        self._fk_v_kdl[finger_idx].JntToCart(self.joints_to_kdl('velocities', joint_velocities,finger_idx),
                                 end_frame, link_idx)
        return end_frame.GetTwist()

    def inverse_kinematics(self, position, orientation=None, seed=None, finger_idx=0):
        pos = PyKDL.Vector(position[0], position[1], position[2])
        if orientation is not None:
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(orientation[1],
                                 orientation[2], orientation[3], orientation[0])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self.n_joints(finger_idx))
        if seed != None:
            seed_array.resize(len(seed))
            for idx, jnt in enumerate(seed):
                seed_array[idx] = jnt
        else:
            seed_array = self.joints_to_kdl('positions')

        # Make IK Call
        if orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)

        result_angles = PyKDL.JntArray(self.n_joints(finger_idx))

        if self._ik_p_kdl[finger_idx].CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = np.array(list(result_angles))
            return result
        else:
            return None

    def jacobian(self, joint_values=None, finger_idx=0):
        jacobian = PyKDL.Jacobian(self._chain_num_joints[finger_idx])
        self._jac_kdl[finger_idx].JntToJac(self.joints_to_kdl('positions', joint_values, finger_idx), jacobian)
        return self.kdl_to_mat(jacobian)

    def jacobian_transpose(self, joint_values=None, finger_idx=0):
        return self.jacobian(joint_values,finger_idx).T

    def jacobian_pseudo_inverse(self, joint_values=None, finger_idx=0):
        return np.linalg.pinv(self.jacobian(joint_values,finger_idx))

    def inertia(self, joint_values=None, finger_idx=0):
        inertia = PyKDL.JntSpaceInertiaMatrix(self.n_joints(finger_idx))
        self._dyn_kdl[finger_idx].JntToMass(self.joints_to_kdl('positions', joint_values), inertia)
        return self.kdl_to_mat(inertia)

    def cart_inertia(self, joint_values=None, finger_idx=0):
        js_inertia = self.inertia(joint_values,finger_idx)
        jacobian = self.jacobian(joint_values,finger_idx)
        return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)


    def n_joints(self, finger_idx = 0):
        return self._chain_num_joints[finger_idx]

    def n_chains(self):
        return self._num_chains

    def n_links(self, finger_idx = 0):

        return self._n_links[finger_idx]

    def base_link(self):
        return self._base_link

    def link_name(self, finger_idx, link_idx):

        chain = self._chains[finger_idx]
        
        return chain.getSegment(link_idx).getName()

    def joint_names(self):
        return self._joint_names