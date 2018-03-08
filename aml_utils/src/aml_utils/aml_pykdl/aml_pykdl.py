import numpy as np
import PyKDL as kdl

class AMLPyKDL(object):

    def __init__(self, kdl_chain, base_position):

        self._chain = kdl_chain
        
        self._num_joints = kdl_chain.getNrOfJoints()

        self._base_position = base_position

        self.setup_solvers()

    def setup_solvers(self):
        
        self._fwd_k = kdl.ChainFkSolverPos_recursive(self._chain)

        self._vel_ik= kdl.ChainIkSolverVel_pinv(self._chain)

        self._pos_ik= kdl.ChainIkSolverPos_NR(self._chain, self._fwd_k, self._vel_ik)

        self._jac   = kdl.ChainJntToJacSolver(self._chain)

    def kdl_to_mat(self, data):
        
        mat =  np.array(np.zeros((data.rows(), data.columns())))

        for i in range(data.rows()):

            for j in range(data.columns()):

                mat[i,j] = data[i,j]

        return mat

    def update_chain(self, q):
        
        self._last_joint_pos = kdl.JntArray(self._num_joints)

        for k in range(self._num_joints):

            self._last_joint_pos[k] = q[k]

    def compute_fwd_kinematics(self, q=None, ori_type='eul'):

        if q is not None:

            assert len(q) == self._num_joints
            
            joint_angles=kdl.JntArray(self._num_joints)
            
            for k in range(self._num_joints):

                joint_angles[k] = q[k]
        else:

            joint_angles =  self._last_joint_pos
         
        final_frame = kdl.Frame()
        
        self._fwd_k.JntToCart(joint_angles, final_frame)

        ee_pos = np.zeros_like(self._base_position)

        for k in range(len(self._base_position)):

            ee_pos[k] = final_frame.p[k] + self._base_position[k]

        if ori_type == 'eul':
            
            rotation = kdl.Rotation(final_frame.M).GetEulerZYX()

            ee_ori = np.asarray([rotation[0], rotation[1], rotation[2]])

        elif ori_type == 'quat':

            rotation = kdl.Rotation(final_frame.M).GetQuaternion()

            ee_ori = np.asarray([rotation[3], rotation[0], rotation[1], rotation[2]])


        return ee_pos, ee_ori

    def compute_inv_kinematics(self, ee_pos):

        desired_frame = kdl.Frame(kdl.Vector(ee_pos[0],ee_pos[1],0))
        
        q_out = kdl.JntArray(self._num_joints)
        
        self._pos_ik.CartToJnt(self._last_joint_pos, desired_frame, q_out)

        return np.asarray(q_out)

    def compute_jacobian(self):
        
        jacobian = kdl.Jacobian(self._num_joints)
        
        self._jac.JntToJac(self._last_joint_pos, jacobian)
        
        jac = self.kdl_to_mat(jacobian)
        
        return jac