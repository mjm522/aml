import copy
from aml_rl_envs.config import AML_RL_ENV_CONFIG
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG

HAND_OBJ_CONFIG = copy.deepcopy(AML_RL_ENV_CONFIG)
HAND_CONFIG = copy.deepcopy(AML_RL_ROBOT_CONFIG)