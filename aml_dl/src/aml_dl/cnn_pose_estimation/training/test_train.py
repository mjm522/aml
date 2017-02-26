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

### Training ###                            position: np.expand_dims(np.ones(3),axis=0)

p0 = np.array([0,0,0])
p1 = np.array([0,1,0])
p2 = np.array([0,1,1])
points = np.concatenate((p0,p1,p2))

for iteration in range(5000):
    with tf.device("/gpu:0"):
        loss = sess.run([loss_op,train_op],feed_dict={input_tensor: np.expand_dims(image,axis=0),
                                     position: np.expand_dims(points,axis=0)
                                    })

        # print every 50 iters
        if iteration%100 == 0:
            print "iteration %d loss: "%(iteration), loss




######## SAVING MODEL #######

save_path = saver.save(sess, "./model.ckpt")
# saver.export_meta_graph("./model.ckpt")
print("Model saved in file: %s" % save_path)
    

sess.close()




