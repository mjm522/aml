import copy
from aml_io.io_tools import get_aml_package_path, get_abs_path

urdf_root_path = get_abs_path(get_aml_package_path()+'/../..') + '/src/aml/aml_rl/aml_rl_envs/src/aml_rl_envs/models/'

WRITE_ENV_CONFIG = {
    'dt':0.01,
    'renders':True,
    'ctrl_type':'pos',
    'max_steps':5000,
    'time_step':1./240.,
    'urdf_root_path':urdf_root_path,
    'cam_dist':1.3,
    'cam_yaw':-180.,
    'cam_pitch':-40,
    'large_val_observation':100,
    'render_height':720,
    'render_width':960,

}