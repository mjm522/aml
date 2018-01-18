import numpy as np
import tensorflow as tf
from aml_opt.policy_opt.reps import REPS
from aml_opt.policy_opt.point_env import PointEnv
from aml_opt.policy_opt.mlp_policy import MLPPolicy
from config import demo_config

np.random.seed(1234)
tf.set_random_seed(1234)

with tf.device('/cpu:0'):
	sess   = tf.Session()
	env    = PointEnv(demo_config['env_config'])
	policy = MLPPolicy(tf_sess=sess, config=demo_config['policy_config'])
	algo   = REPS(env=env, tf_sess=sess, policy=policy, config=demo_config['algo_config'])
	algo.init_opt()
	algo.train()