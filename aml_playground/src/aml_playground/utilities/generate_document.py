import os
import numpy as np
from pylatex.utils import italic, NoEscape
from pylatex import Document, Math, Section, Subsection, Tabular, Tabularx, Command, Figure, SubFigure, NoEscape, Itemize, Alignat

class GenerateDocument():

    def __init__(self, config):

        geometry_options = {
        "head": "10pt",
        "margin": "0.2in",
        "bottom": "0.2in",
        "includeheadfoot": False
        }
        
        self._doc = Document(config['file_name'], geometry_options=geometry_options)
        self._doc.preamble.append(Command('title', config['title']))
        self._doc.preamble.append(Command('author', config['authors']))
        self._doc.preamble.append(Command('date', NoEscape(r'\today')))
        self._doc.append(NoEscape(r'\maketitle'))

    def add_image(self, image_filenames, image_captions):

        for k in range(2):
  
            with self._doc.create(Figure(position='h!')) as image:

                with self._doc.create(SubFigure(
                        position='b',
                        width=NoEscape(r'0.5\linewidth'))) as left_image:

                    left_image.add_image(image_filenames[k][0],
                                          width=NoEscape(r'\linewidth'))
                    left_image.add_caption(image_captions[k][0])

                with self._doc.create(SubFigure(
                        position='b',
                        width=NoEscape(r'0.5\linewidth'))) as right_image:

                    right_image.add_image(image_filenames[k][1],
                                           width=NoEscape(r'\linewidth'))
                    right_image.add_caption(image_captions[k][1])


    def add_table(self, table_values=None):

        table_rows = ['Goal pos weight', 'Goal vel weight', 'Control weight','Delta Ctrl weight', 'Kp-Kd weight','Delta Kp-Kd weight',
                    'Cumsum reward', 'Sigmoid reward','Gamma','Next force prediction','Time step','Kd Scale']


        with self._doc.create(Subsection('Table of Hyperparameters')):

            with self._doc.create(Tabular(table_spec='|l|c||',
                                         row_height=1.4)) as table:

                for k in range(len(table_rows)):
                    table.add_hline()
                    table.add_row((table_rows[k], table_values[k]))
                    table.add_hline()


    def fill_document(self):

        with self._doc.create(Section('Point-Mass object pulling a spring (pybullet)')):
            
            self._doc.append('Task: Learn the varying 3D Kp and Kd required to pull a spring to a desired height (with minimum effort)')
            with self._doc.create(Itemize()) as itemize:
                itemize.add_item("100 time steps ( Hence, 100 x (3 + 3) parameters to learn as single policy or 100 separate policies )")
                itemize.add_item("Learner: CREPS")
                itemize.add_item("Policy: Linear Gaussian")

            self._doc.append('Other Info:')
            with self._doc.create(Itemize()) as itemize:
                itemize.add_item("Time steps = 100")
                itemize.add_item("Spring stiffness = 3")
                itemize.add_item("Episodes = 1000")
                itemize.add_item("Kp scale = 0.25*(Kd scale^2)")

            self._doc.append('REPS params:')
            with self._doc.create(Itemize()) as itemize:
                itemize.add_item("Entropy bound = 2.0")
                itemize.add_item("Context dim = 9 (3x pos, 3x delta_pos, 3x force)")
                itemize.add_item("Context feature dim = 9")
                itemize.add_item("1 policy per time step")

            self._doc.append("Cost function")
            with self._doc.create(Alignat(numbering=False, escape=False)) as agn:
                agn.append(r'\sum_t \left( X^T_t Q_1 X_t + \dot{X}_t^T Q_2 \dot{X}_t + U^T_t R_1 U_t + \dot{U}_t^T R_2 \dot{U}_t + K^T_t R_3 K_t + \dot{K_t}^T R_4 \dot{K_t} \right)')

    def finish(self):

        self._doc.generate_pdf(clean_tex=False)


def create_experiment_document(exp_name, image_folder, exp_params):

    exp_dir = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/point_mass/hyper_param_search/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    config = {'file_name':exp_dir+exp_name,
    'title':'Variable Impedance Learning',
    'authors':'Mike and Saif'}

    gen_doc = GenerateDocument(config)
    gen_doc.fill_document()

    
    image_filenames = [ [image_folder+'Kp-Kd-u.png', image_folder+'Traj-Vel.png'],
                      [image_folder+'Reward.png', image_folder+'Reward_traj.png',],
                      ]
    image_captions = [['Param values', 'Trajectory and Velocity'],['Mean reward', 'Reward splits']]

    '''
    ['Goal pos weight', 'Goal vel weight', 'Control weight','Delta Ctrl weight', 'Kp-Kd weight','Delta Kp-Kd weight',
                    'Enable cumsum reward', 'Enable sigmoid reward','Gamma','Next force prediction','Time step','Kd Scale']
    '''

    if exp_params['env_params']['force_predict_model'] is not None:
        force_predict_model = True

    table_values=[exp_params['env_params']['goal_pos_weight'][0,0],
                  exp_params['env_params']['goal_vel_weight'][0,0],
                  exp_params['env_params']['u_weight'][0,0],
                  exp_params['env_params']['delta_u_weight'][0,0],
                  exp_params['env_params']['param_weight'][0,0],
                  exp_params['env_params']['delta_param_weight'][0,0],
                  exp_params['env_params']['enable_cumsum'],
                  exp_params['env_params']['enable_sigmoid'],
                  exp_params['env_params']['reward_gamma'],
                  force_predict_model,
                  exp_params['env_params']['time_step'],
                  exp_params['env_params']['param_scale'][-1],
                  ]

    gen_doc.add_table(table_values=table_values)

    gen_doc.add_image(image_filenames=image_filenames, image_captions=image_captions)

    gen_doc.finish()



if __name__ == '__main__':
    
    from aml_playground.var_imp_reps.exp_params.experiment_point_mass_params import exp_params
    
    create_experiment_document(exp_name='trial', 
                               image_folder=os.environ['AML_DATA'] + '/aml_playground/imp_worlds/point_mass/',
                               exp_params=exp_params)
