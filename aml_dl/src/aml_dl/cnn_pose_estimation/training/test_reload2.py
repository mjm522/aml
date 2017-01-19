import os, sys

aml_dl_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..', ''))
sys.path.append(aml_dl_path)

import cnn_pose_estimation.model.tf_model as tf_model
from cnn_pose_estimation.training.config import network_params

import numpy as np
import tensorflow as tf

np.random.seed(seed=42)

def fakeImageInput(network_params):
    random_image = np.random.randn(network_params['image_height'],network_params['image_width'],network_params['image_channels'])
    image = np.transpose(random_image,(2,1,0)) # If the image is WxHxC, make it CxWxH
    image = image.flatten()

    return image, random_image

nn = tf_model.multi_modal_network_fp(dim_input=network_params['image_size']+32,
                                      network_config=network_params)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

fc_op = nn['fc_pose_out']
# fc_action = nn['fc_action_out']
features_op = nn['features']
loss_op = nn['loss_pose']
input_tensor = nn['nn_input']
position = nn['pose']
train_op = tf_model.train_adam_step(loss_op)

init_fn = nn['init_fn']


# Initialialise session and variables

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
init_fn(sess)

# Run a simple test with a fake image
image, random_image = fakeImageInput(network_params)
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

output = sess.run([fc_op,loss_op,features_op], feed_dict={input_tensor: np.expand_dims(np.r_[np.zeros(32),image],axis=0),
                                 						position: np.expand_dims(points,axis=0),
                                 						# state_input: np.expand_dims(state,axis=0),
                                 						# action: np.expand_dims(action_groundtruth,axis=0),
                                 						# precision: np.expand_dims(prc,axis=0),
                                })

print "fc_output:", output[0]
print "loss_output:", output[1]
print "features:", output[2]
# print "fc2_output:", output[2]
# print "loss_fc2:", output[3]


sess.close()




