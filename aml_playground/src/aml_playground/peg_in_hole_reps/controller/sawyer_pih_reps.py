import os
import numpy as np
from rl_algos.agents.gpreps import GPREPSOpt
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.sawyer.sawyer_peg_env import SawyerEnv
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

np.random.seed(123)

class SawyerPegREPS():

    def __init__(self, joint_space):

        self._env = SawyerEnv()

        kwargs = {}
        kwargs['limb_name'] = 'right' 

        path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/right_sawyer_exp_peg_in_hole_01.pkl'

        if not os.path.exists(path_to_demo):
            raise Exception("Enter a valid demo path")
        else:
            kwargs['path_to_demo'] = path_to_demo

        if joint_space:
            self._gen_traj = JSTrajGenerator(load_from_demo=True, **kwargs)
        else:
            self._gen_traj = OSTrajGenerator(load_from_demo=True, **kwargs)

        self._demo_traj = self._gen_traj.generate_traj()['pos_traj']

        # plot_demo(self._demo_traj.T)

        self._dof = self._demo_traj.shape[1]

        self.encode_dmps()

    def encode_dmps(self):

        discrete_dmp_config['dof'] = self._dof

        self._man_dmp = {}
        self._man_dmp['config'] = discrete_dmp_config
        self._man_dmp['obj'] = DiscreteDMP(config=discrete_dmp_config)

        self._man_dmp['obj'].load_demo_trajectory(self._demo_traj)
        self._man_dmp['obj'].train()

    def update_dmp_params(self, phase_start=1., speed=1., goal_offset=None, start_offset=None, external_force=None):

        if goal_offset is None: goal_offset = np.zeros(self._dof)

        if start_offset is None: start_offset = np.zeros(self._dof)

        dmp    = self._man_dmp['obj']
        config = self._man_dmp['config']

        config['y0'] = dmp._traj_data[0, 1:] + start_offset
        config['dy'] = np.zeros(self._dof)
        config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
        config['tau'] = 1./speed
        config['phase_start'] = phase_start

        if external_force is None:
            external_force = np.zeros(self._dof)
            config['type'] = 1
        else:
            config['type'] = 3

        config['extForce'] = external_force

        new_dmp_traj = dmp.generate_trajectory(config=config)

        return new_dmp_traj['pos']
