import numpy as np
import pybullet as pb
from os.path import join
from aml_rl_envs.point_mass.config import POINT_MASS_CONFIG
from aml_robot.bullet.bullet_visualizer import setup_bullet_visualizer

class PointMass():

    def __init__(self, cid, config=POINT_MASS_CONFIG, scale=1., pos=(0.,0.,0.53), ori=(0.,0.,0.,1.)):

        self._config = config

        self._ee_index = -1

        self._scale = scale

        self._base_pos = pos

        self._base_ori = ori

        self._ft_sensor_jnt = self._ee_index

        if cid is None:
            self._cid = setup_bullet_visualizer(True)
        else:
            self._cid = cid

        self.reset()
 
    def reset(self, jnt_pos = None):

        self._robot_id = pb.loadURDF(join(self._config['urdf_root_path'],"box.urdf"), globalScaling=self._scale, 
                                                              useFixedBase=False, physicsClientId=self._cid)

        pb.resetBasePositionAndOrientation(self._robot_id, self._base_pos, self._base_ori, physicsClientId=self._cid)

        tmp_id = pb.loadURDF(join(self._config['urdf_root_path'],"box2.urdf"), globalScaling=self._scale, 
                                                                  useFixedBase=False, physicsClientId=self._cid)

        pb.resetBasePositionAndOrientation(tmp_id, (0.,0.,1.03), self._base_ori, physicsClientId=self._cid)


    def simple_step(self):
        pb.stepSimulation(physicsClientId=self._cid)


    def state(self, ori_type='quat'):

        ee_pos, ee_ori = self.get_ee_pose(as_tuple=True)

        ee_vel, ee_omg = self.get_ee_velocity()

        # jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = pb.getLinkState(idx=self._ee_index)

        state = {}
        state['effort'] = np.array([0.,0.,0.]) #np.asarray(jnt_reaction_forces)
        state['ee_point'] = np.asarray(ee_pos)
        
        if ori_type != 'quat':
            state['ee_ori'] = np.asarray(pb.getEulerFromQuaternion(ee_ori))
        else:
            state['ee_ori'] = np.array([ee_ori[3],ee_ori[0],ee_ori[1],ee_ori[2]])

        state['ee_vel'] = ee_vel
        state['ee_omg'] = ee_omg

        return state


    def apply_action(self, u):

        pb.applyExternalForce(objectUniqueId=self._robot_id,
                              linkIndex=-1,
                              forceObj=u,
                              posObj=self.get_ee_pose()[0],
                              flags=pb.WORLD_FRAME,
                              physicsClientId=self._cid)

    def get_ee_pose(self, as_tuple=False):

        link_state = pb.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._cid)

        ee_pos = link_state[0]
        ee_ori = link_state[1]

        if not as_tuple:
            ee_pos = np.asarray(ee_pos) 
            ee_ori = np.asarray(ee_ori)

        return ee_pos, ee_ori

    def get_ee_velocity(self):

        vel = pb.getBaseVelocity(self._robot_id, physicsClientId=self._cid)

        return np.asarray(vel[0]), np.asarray(vel[1])

    def get_ee_wrench(self, local=False):
        '''
            End effector forces and torques.
            Returns [fx, fy, fz, tx, ty, tz]
        '''

        jnt_reaction_force = np.array([0.,0.,0,0.,0.,0.])
        # _, _,jnt_reaction_force, _ = self.get_jnt_state(idx=self._ft_sensor_jnt)

        if local:
            
            link_state = pb.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._cid)

            ee_pos = link_state[0]
            ee_ori = link_state[1]

            jnt_reaction_force = np.asarray(jnt_reaction_force)
            force  = tuple(jnt_reaction_force[:3])
            torque = tuple(jnt_reaction_force[3:])

            inv_ee_pos, inv_ee_ori = pb.invertTransform(ee_pos, ee_ori)
            
            force, _  = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, force, (0,0,0,1))
            torque, _ = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, torque, (0,0,0,1))
            jnt_reaction_force = force + torque

        return jnt_reaction_force


def main():

    point_mass = PointMass(cid=None, scale=0.1)

    while True:
        point_mass.simple_step()
        print "Pos \t", point_mass.get_ee_pose()[0]
        
        print "Vel \t", point_mass.get_ee_velocity()[0]

        point_mass.apply_action(u=np.array([0.,0.,0.02]))


if __name__ == '__main__':
    main()