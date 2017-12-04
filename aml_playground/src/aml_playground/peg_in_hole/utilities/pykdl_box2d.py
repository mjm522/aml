import numpy as np
from PyKDL import *
from aml_playground.peg_in_hole.pih_worlds.box2d.config import man_config


class PyKDLBox2d(object):

    def __init__(self, config):
        self._config = config
        self._chain  = Chain()
        self._base_position = self._config['joints'][0]['anchor']
        self._num_joints = len(self._config['joints'])
        self._fwd_k  = None
        self.setup_chain()

    def setup_chain(self):

        for k in range (1,len(self._config['joints'])):
            prev_joint_pos = self._config['joints'][k-1]['anchor']
            joint_pos = self._config['joints'][k]['anchor']
            #add a segment to the pykdl
            #frame is a relative frame from the previous joint
            self._chain.addSegment(Segment(Joint(Joint.RotZ),Frame(Vector(joint_pos[0]-prev_joint_pos[0],joint_pos[1]-prev_joint_pos[1],0.))))

        last_link_dim = self._config['links'][-1]['dim']
        last_jnt_pos  = self._config['joints'][-1]['anchor']

        ee_pos = (last_jnt_pos[0], last_jnt_pos[1]+2*last_link_dim[1])

        self._chain.addSegment(Segment(Joint(Joint.RotZ),Frame(Vector(ee_pos[0],ee_pos[1],0.))))
        self._fwd_k = ChainFkSolverPos_recursive(self._chain)
        self._vel_ik= ChainIkSolverVel_pinv(self._chain)
        self._pos_ik= ChainIkSolverPos_NR(self._chain, self._fwd_k, self._vel_ik)
        self._jac   = ChainJntToJacSolver(self._chain)

    def kdl_to_mat(self, data):
        mat =  np.array(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i,j] = data[i,j]
        return mat

    def update_chain(self, q):
        self._last_joint_pos = JntArray(self._num_joints)
        for k in range(self._num_joints):
            self._last_joint_pos[k] = q[k]

    def compute_fwd_kinematics(self, q=None):

        joint_angles=JntArray(self._num_joints)
        
        if q is not None:
            assert len(q) == self._num_joints
            for k in range(self._num_joints):
                joint_angles[k] = q[k]
        else:
            joint_angles =  self._last_joint_pos
         
        final_frame = Frame()
        self._fwd_k.JntToCart(joint_angles, final_frame)
        final_ee = (final_frame.p[0] + self._base_position[0], final_frame.p[1] + self._base_position[1])

        rotation = Rotation(final_frame.M).GetEulerZYX()

        return np.asarray([final_ee[0], final_ee[1], rotation[0]])

    def compute_inv_kinematics(self, ee_pos):

        desired_frame = Frame(Vector(ee_pos[0],ee_pos[1],0))
        q_out = JntArray(self._num_joints)
        self._pos_ik.CartToJnt(self._last_joint_pos, desired_frame, q_out)

        return np.asarray(q_out)

    def compute_jacobian(self):
        jacobian = Jacobian(self._num_joints)
        self._jac.JntToJac(self._last_joint_pos, jacobian)
        jac = self.kdl_to_mat(jacobian)
        return np.vstack([jac[0,:], jac[1,:], jac[5,:]])


def main():
    pb = PyKDLBox2d(man_config)
    q = [np.pi/4,0.,0]
    ee_pos = [-8.07106781,  7.32106781]
    print "Forward Kinematics"
    print pb.fwd_kinematics(q=q)
    pb.update_chain(q)
    print "Inverse Kinematics"
    print pb.inv_kinematics(ee_pos)
    print pb.jacobian()


if __name__ == '__main__':
    main()