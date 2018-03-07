import numpy as np
import PyKDL as kdl

class AMLPyKDL(object):

    def __init__(self, kdl_chain, num_joints, base_position):

        self._num_joints = num_joints

        self._base_position = base_position

        self.setup_solvers()

    def setup_solvers(self):

        for k in range (1,len(self._config['joints'])):
            prev_joint_pos = self._config['joints'][k-1]['anchor']
            joint_pos = self._config['joints'][k]['anchor']
            #add a segment to the pykdl
            #frame is a relative frame from the previous joint
            self._chain.addSegment(kdl.Segment(kdl.Joint(Joint.RotZ), kdl.Frame( kdl.Vector(joint_pos[0]-prev_joint_pos[0],joint_pos[1]-prev_joint_pos[1],0.))))

        last_link_dim = self._config['links'][-1]['dim']

        ee_pos = (0, 2*last_link_dim[1])
        self._chain.addSegment(kdl.Segment(kdl.Joint(Joint.RotZ), kdl.Frame(kdl.Vector(ee_pos[0],ee_pos[1],0.))))
        
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

    def compute_fwd_kinematics(self, q=None):

        if q is not None:
            assert len(q) == self._num_joints
            
            joint_angles=kdl.JntArray(self._num_joints)
            
            for k in range(self._num_joints):
                joint_angles[k] = q[k]
        else:
            joint_angles =  self._last_joint_pos
         
        final_frame = kdl.Frame()
        self._fwd_k.JntToCart(joint_angles, final_frame)
        final_ee = (final_frame.p[0] + self._base_position[0], final_frame.p[1] + self._base_position[1])

        rotation = kdl.Rotation(final_frame.M).GetEulerZYX()

        return np.asarray([final_ee[0], final_ee[1], rotation[0]])

    def compute_inv_kinematics(self, ee_pos):

        desired_frame = kdl.Frame(Vector(ee_pos[0],ee_pos[1],0))
        q_out = kdl.JntArray(self._num_joints)
        self._pos_ik.CartToJnt(self._last_joint_pos, desired_frame, q_out)

        return np.asarray(q_out)

    def compute_jacobian(self):
        jacobian = kdl.Jacobian(self._num_joints)
        self._jac.JntToJac(self._last_joint_pos, jacobian)
        jac = self.kdl_to_mat(jacobian)
        return np.vstack([jac[0,:], jac[1,:], jac[5,:]])