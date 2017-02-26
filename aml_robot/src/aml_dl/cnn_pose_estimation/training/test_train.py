import os, sys

aml_dl_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..', ''))
sys.path.append(aml_dl_path)

import cnn_pose_estimation.model.tf_model as tf_model
from cnn_pose_estimation.training.config import network_params

import numpy as np
import tensorflow as tf

# Fixing seed
np.random.seed(seed=42)

def fakeImageInput(network_params):
    image = np.random.randn(network_params['image_height'],network_params['image_width'],network_params['image_channels'])
    image = np.transpose(image,(2,1,0)) # If the image is WxHxC, make it CxWxH
    image = image.flatten()

    return image

nn = tf_model.pose_estimation_network(dim_input=network_params['image_size'],
                                      network_config=network_params)

# Add ops to save and restore all the variables.


fc_op = nn['fc_out']
features_op = nn['features']
loss_op = nn['loss']
input_tensor = nn['input']
position = nn['position']
train_op = tf_model.train_adam_step(loss_op)

saver = tf.train.Saver()


# Initialialise session and variables

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

# Run a simple test with a fake image
image = fakeImageInput(network_params)

### Training ###
input_image = np.expand_dims(image,axis=0)
print input_image
for iteration in range(4000):
    with tf.device("/cpu:0"):
        
        loss = sess.run([loss_op,train_op,fc_op],feed_dict={input_tensor: input_image,
                                     position: np.expand_dims(np.ones(3),axis=0)
                                    })

        # print every 50 iters
        if iteration%100 == 0:
            print "iteration %d loss: "%(iteration), loss




######## SAVING MODEL #######

save_path = saver.save(sess, "./model.ckpt")
# saver.export_meta_graph("./model.ckpt")
print("Model saved in file: %s" % save_path)
    

sess.close()




