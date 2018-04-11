import copy
from aml_rl_envs.config import AML_RL_ENV_CONFIG
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG

BAXTER_ENV_CONFIG = copy.deepcopy(AML_RL_ENV_CONFIG)
BAXTER_CONFIG = copy.deepcopy(AML_RL_ROBOT_CONFIG)

BAXTER_ENV_CONFIG['is_discrete'] = False
BAXTER_ENV_CONFIG['action_repeat'] = 1