import tensorflow as tf




def optimiser_op(loss_op, optimiser_params):


	if optimiser_params['type'] == 'adam':

		tmp = tf.train.AdamOptimizer(learning_rate=optimiser_params['params']['learning_rate'], 
									  beta1=optimiser_params['params']['beta1'],
									  beta2=optimiser_params['params']['beta2'],
									  epsilon = optimiser_params['params']['epsilon'],
									  use_locking = optimiser_params['params']['use_locking'])

		return tmp.minimize(loss_op)
	else:
		raise ValueError("Unknown optimiser %s"%(optimiser_params['type'],))