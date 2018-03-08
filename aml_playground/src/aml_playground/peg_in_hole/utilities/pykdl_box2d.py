import numpy as np
import PyKDL as kdl
from aml_pykdl import AMLPyKDL 
from aml_playground.peg_in_hole.pih_worlds.box2d.config import man_config


class PyKDLBox2d(AMLPyKDL):

    def __init__(self, config):

        self._config = config

        self._base_position = self._config['joints'][0]['anchor']

        self.setup_chain()

        AMLPyKDL.__init__(self, self.setup_chain(), self._base_position)

    def setup_chain(self):

        chain  = kdl.Chain()

        for k in range (1,len(self._config['joints'])):
            
            prev_joint_pos = self._config['joints'][k-1]['anchor']
            
            joint_pos = self._config['joints'][k]['anchor']
            #add a segment to the pykdl
            #frame is a relative frame from the previous joint

            chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame(kdl.Vector(joint_pos[0]-prev_joint_pos[0],joint_pos[1]-prev_joint_pos[1],0.))))

        last_link_dim = self._config['links'][-1]['dim']

        ee_pos = (0, 2*last_link_dim[1])

        chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame(kdl.Vector(ee_pos[0],ee_pos[1],0.))))

        return chain

    def get_ee_pose(self, q):
        
        ee_pos, ee_ori = self.compute_fwd_kinematics(q=q)
        
        return np.asarray([ee_pos[0], ee_pos[1], ee_ori[0]])

    def get_jacobian(self):

        jac = self.compute_jacobian()

        return np.vstack([jac[0,:], jac[1,:], jac[5,:]])

def main():
    pb = PyKDLBox2d(man_config)
    q = [np.pi/4,0.,0]
    ee_pos = [-8.07106781,  7.32106781]
    print "Forward Kinematics"
    print pb.get_ee_pose(q=q)
    pb.update_chain(q)
    print "Inverse Kinematics"
    print pb.compute_inv_kinematics(ee_pos)
    print pb.get_jacobian()


if __name__ == '__main__':
    main()