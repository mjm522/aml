import os
import numpy as np
from aml_opt.policy_opt.gpreps import REPS
from aml_io.io_tools import load_data
from aml_playground.peg_in_hole.pih_worlds.box2d.box2d_pih_world import Box2DPIHWorld
from aml_playground.peg_in_hole.pih_worlds.box2d.config import pih_world_config as config


config = {
    'epsilon':0.5,
    'L2_reg_dual'=0.,  # 1e-5,
    'L2_reg_loss'=0.,
    'max_opt_itr'=50,
    'max_pol_itr'=50,
    'N':150,
    'T':100,
    's_dim':3,
    'a_dim':3,
}


class FwdModel():

    def __init__(self, model, demo, config=config):
        self._model  =  model
        self._config = config
        self._demo   = demo

    def compute_cmd(self, set_point, gain):
        return self._model.compute_os_ctrlr_cmd(os_set_point=set_point, Kp=gain)

    def update(self):
        self._model.update()

    def simulate(self, k, gain):
        Kp = numpy.random.multivariate_normal(mean=gain[:self._config['u_dim']], cov=np.diag(gain[self._config['u_dim']:]))
        action = self._fwd_model.compute_cmd(set_point=self._demo[k,:], gain=Kp)
        self._fwd_model.update(action)

        return self._fwd_model.get_state()['ee_pos']



class Cost():
    
    def __init__(self, demo, fwd_model, config=config):
        self._fwd_model = fwd_model
        self._demo      =  demo
        self._config    = config

    def running_cost(self, k, ee_pos):
        return 0.5*np.linalg.norm(ee_pos-self._demo[k,:])

    def evaluate_policy(self, policy):
        trajectory_cost = np.zeros(len(policy))

        self._fwd_model.set_joint_pos([0., 0., 0.])

        for k in range(len(demo)):
            Kp = numpy.random.multivariate_normal(mean=policy[k][:self._config['u_dim']], cov=np.diag(policy[k][self._config['u_dim']:]))
            action = self._fwd_model.compute_cmd(set_point=self._demo[k,:], gain=Kp)
            self._fwd_model.update(action)
            trajectory_cost[k] = self.running_cost(k, self._fwd_model.get_state()['ee_pos'])

        return trajectory_cost


def load_demo():
    data = load_data(os.environ['AML_DATA'] + '/aml_playground/pih_worlds/box2d/man_data.pkl')
    demo = []
    for state in data:
        demo.append(state['set_point'])

    return np.asarray(demo)



def main():

    demo = load_demo()

    world = Box2DPIHWorld(config)

    fwd_model = FwdModel(model=world._manipulator)
    cost_fn   = Cost(demo=demo, fwd_model=fwd_model)

    reps = REPS(demo=demo,
                cost=cost_fn, 
                fwd_model=fwd_model, 
                config=config)
    reps.run()


if __name__ == '__main__':
    main()