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


# Initialialise session and variables

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

# Run a simple test with a fake image
image = fakeImageInput(network_params)

######## RESTORING MODEL #######
saver.restore(sess, "model.ckpt")
print("Model restored.")
# Do some work with the model
output = sess.run([fc_op,loss_op], feed_dict={input_tensor: np.expand_dims(image,axis=0),
                                 position: np.expand_dims(np.ones(3),axis=0)
                                })

print "fc_output:", output[0]
print "loss_output:", output[1]


sess.close()




