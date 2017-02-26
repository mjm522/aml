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
sess = tf.Session()

# saver = tf.train.Saver()
# saver = tf.train.import_meta_graph('./model.ckpt-0.meta')
# ######## RESTORING MODEL #######
# saver.restore(sess, "./model.ckpt-0")
# print("Model restored.")
saver = tf.train.import_meta_graph('model.ckpt.meta')
ckpt = tf.train.get_checkpoint_state('./')
print ckpt
print "Latest: ", tf.train.latest_checkpoint('./')
if ckpt and ckpt.model_checkpoint_path:
    print(ckpt.model_checkpoint_path)
    saver.restore(sess,  tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()

fc_op = nn['fc_out']
features_op = nn['features']
loss_op = nn['loss']
input_tensor = nn['input']
position = nn['position']



train_op = tf_model.train_adam_step(loss_op)


all_vars = tf.trainable_variables()
for v in all_vars:
    print(v.name)

# with tf.variable_scope('action_net'):
#     dim_output = 7
#     n_layers = 3
#     layer_size = 20
#     dim_hidden = (n_layers - 1)*[layer_size]
#     dim_hidden.append(dim_output)

#     state_input = tf.placeholder("float", [None, 27], name='nn_state_input')
#     precision = tf.placeholder("float", [None, dim_output,dim_output], name='precision')
#     action = tf.placeholder("float", [None, dim_output], name='action')

#     fc2_input = tf.concat(concat_dim=1, values=[features_op, state_input])

#     fc2_output, weights_FC2, biases_FC2 = tf_model.get_mlp_layers(fc2_input, n_layers, dim_hidden)
#     fc2_vars = weights_FC2 + biases_FC2
#     loss_fc2 = tf_model.euclidean_loss_layer(a=action, b=fc2_output, precision=precision, batch_size=25)



# Initialialise session and variables

init_op = tf.global_variables_initializer()

sess.run(init_op)

# Run a simple test with a fake image
image = fakeImageInput(network_params)

input_image = np.expand_dims(image,axis=0)

print input_image

# Do some work with the model
output = sess.run([fc_op,loss_op], feed_dict={input_tensor: input_image,
                                 position: np.expand_dims(np.ones(3),axis=0)
                                })

print "fc_output:", output[0]
print "loss_output:", output[1]
# print "Trainable vars:", tf.trainable_variables()

sess.close()




