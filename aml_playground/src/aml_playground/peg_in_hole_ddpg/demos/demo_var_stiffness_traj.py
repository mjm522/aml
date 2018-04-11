import os
from aml_playground.peg_in_hole.exp_params.experiment_params import ee_config as config
from aml_playground.peg_in_hole.controller.variable_stiffness_traj import VaribaleStiffnessFinder

def main():

    vsf = VaribaleStiffnessFinder(config=config)
    vsf.run()

if __name__ == '__main__':
    main()