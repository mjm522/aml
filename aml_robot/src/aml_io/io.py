
import pickle
import tensorflow as tf




def save_data(data, filename):

    output = open(filename, 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)

    # Pickle the list using the highest protocol available.
    # pickle.dump(selfref_list, output, -1)

    output.close()


def load_data(filename):
	
    try:
        pkl_file = open(filename, 'rb')
    except Exception as e:
        raise e

    data = pickle.load(pkl_file)

    pkl_file.close()

    return data


def load_tf_check_point(session, filename):

    saver = tf.train.Saver()

    saver.restore(session, filename)

    print("Model restored.")


def save_tf_check_point(session, filename):
    
    saver = tf.train.Saver()

    save_path = saver.save(session, filename)

    print("tf checkpoint saved in file: %s" % save_path)
