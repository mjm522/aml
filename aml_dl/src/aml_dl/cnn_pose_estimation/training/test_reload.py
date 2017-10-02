import os, sys

aml_dl_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..', ''))
sys.path.append(aml_dl_path)

import cnn_pose_estimation.model.tf_model as tf_model
from cnn_pose_estimation.training.config import network_params

import numpy as np
import tensorflow as tf

np.random.seed(seed=42)

def fakeImageInput(network_params):
    image = np.random.randn(network_params['image_height'],network_params['image_width'],network_params['image_channels'])
    image = np.transpose(image,(2,1,0)) # If the image is WxHxC, make it CxWxH
    image = image.flatten()

    return image

nn = tf_model.pose_estimation_network(dim_input=network_params['image_size'],
                                      network_config=network_params)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

fc_op = nn['fc_out']
features_op = nn['features']
loss_op = nn['loss']
input_tensor = nn['input']
position = nn['position']
train_op = tf_model.train_adam_step(loss_op)

with tf.variable_scope('action_net'):
    dim_output = 7
    n_layers = 3
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)

    state_input = tf.placeholder("float", [None, 27], name='nn_state_input')
    precision = tf.placeholder("float", [None, dim_output,dim_output], name='precision')
    action = tf.placeholder("float", [None, dim_output], name='action')

    fc2_input = tf.concat(axis=1, values=[features_op, state_input])

    fc2_output, weights_FC2, biases_FC2 = tf_model.get_mlp_layers(fc2_input, n_layers, dim_hidden)
    fc2_vars = weights_FC2 + biases_FC2
    loss_fc2 = tf_model.euclidean_loss_layer(a=action, b=fc2_output, precision=precision, batch_size=25)



# Initialialise session and variables

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

# Run a simple test with a fake image
image = fakeImageInput(network_params)
state = np.random.randn(27)
action_groundtruth = np.random.randn(7)
prc = np.random.randn(7,7)

######## RESTORING MODEL #######
saver.restore(sess, "model.ckpt")
print("Model restored.")
# Do some work with the model

p0 = np.array([0,0,0])
p1 = np.array([0,1,0])
p2 = np.array([0,1,1])
points = np.concatenate((p0,p1,p2))

output = sess.run([fc_op,loss_op,fc2_output,loss_fc2], feed_dict={input_tensor: np.expand_dims(image,axis=0),
                                 						position: np.expand_dims(points,axis=0),
                                 						state_input: np.expand_dims(state,axis=0),
                                 						action: np.expand_dims(action_groundtruth,axis=0),
                                 						precision: np.expand_dims(prc,axis=0),
                                })

print "fc_output:", output[0]
print "loss_output:", output[1]
print "fc2_output:", output[2]
print "loss_fc2:", output[3]


sess.close()




