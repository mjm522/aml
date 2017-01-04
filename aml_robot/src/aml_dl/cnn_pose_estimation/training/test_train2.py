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
    random_image = np.random.randn(network_params['image_height'],network_params['image_width'],network_params['image_channels'])
    image = np.transpose(random_image,(2,1,0)) # If the image is WxHxC, make it CxWxH
    image = image.flatten()

    return image, random_image

nn = tf_model.multi_modal_network_fp(dim_input=(network_params['image_size']+32),
                                      network_config=network_params)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

fc_op = nn['fc_pose_out']
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

### Training ###

p0 = np.array([0,0,0])
p1 = np.array([0,1,0])
p2 = np.array([0,1,1])
points = np.concatenate((p0,p1,p2))


print "IMAGE SHAPE WHAAT:", image.shape


for iteration in range(1000):
    with tf.device("/cpu:0"):
        loss = sess.run([loss_op,train_op],feed_dict={input_tensor: np.expand_dims(np.r_[np.zeros(32),image],axis=0),
                                     position: np.expand_dims(points,axis=0)
                                    })

        # print every 50 iters
        if iteration%50 == 0:
            print "iteration %d loss: "%(iteration), loss


######## SAVING MODEL #######
save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)
    

sess.close()








# sess.close()




