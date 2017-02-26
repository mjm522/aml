import tensorflow as tf


def load_tf_check_point(session, filename):

    saver = tf.train.Saver()

    saver.restore(session, filename)

    print("Model restored.")


def save_tf_check_point(session, filename):
    
    saver = tf.train.Saver()

    save_path = saver.save(session, filename)

    print("tf checkpoint saved in file: %s" % save_path)