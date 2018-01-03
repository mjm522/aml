import numpy as np
import tensorflow as tf
from aml_opt.policy_opt.reps import REPS
from aml_opt.policy_opt.point_env import PointEnv

from aml_opt.policy_opt.gaussian_mlp_policy import GaussianMLPPolicy
from aml_opt.policy_opt.linear_feature_baseline import LinearFeatureBaseline

np.random.seed(1234)
tf.set_random_seed(1234)

with tf.device('/cpu:0'):
	sess = tf.Session()
	env    = PointEnv()
	policy = GaussianMLPPolicy(env=env, tf_sess=sess)
	algo = REPS(env=env, tf_sess=sess, policy=policy, baseline=LinearFeatureBaseline())
	algo.init_opt()
	algo.train()
