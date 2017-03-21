import os
import numpy as np
import tensorflow as tf
from aml_io.tf_io import load_tf_check_point
from aml_dl.mdn.model.tf_models import tf_model
from aml_dl.utilities.tf_summary_writer import TfSummaryWriter

class NNPushFwdModel(object):

    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._device = self._params['device']

        self._tf_sumry_wrtr = None

        self._optimiser = network_params['optimiser']

        self._data_configured = False

        if network_params['write_summary']:
            if 'summary_dir' in network_params:
                summary_dir = network_params['summary_dir']
            else:
                summary_dir = None
            
            self._tf_sumry_wrtr = TfSummaryWriter(tf_session=sess,summary_dir=summary_dir)
            cuda_path = '/usr/local/cuda/extras/CUPTI/lib64'

            curr_ld_path = os.environ["LD_LIBRARY_PATH"]

            if not cuda_path in curr_ld_path.split(os.pathsep):
                print "Enviroment variable LD_LIBRARY_PATH does not contain %s"%cuda_path
                print "Please add it, else the program will crash!"
                raw_input("Press Ctrl+C")
                # os.environ["LD_LIBRARY_PATH"] = curr_ld_path + ':'+cuda_path

        with tf.device(self._device):
            self._net_ops = tf_model(dim_input=network_params['dim_input'],
                                     dim_output=network_params['dim_output'],
                                     loss_type='quadratic',
                                     cnn_params=network_params['cnn_params'], 
                                     fc_params=network_params['fc_params'],
                                     optimiser_params=network_params['optimiser'],
                                     tf_sumry_wrtr=self._tf_sumry_wrtr)

            self._init_op = tf.initialize_all_variables()

            self._saver = tf.train.Saver()

    def init_model(self):

        with tf.device(self._device):
            self._sess.run(self._init_op)

            if self._params['load_saved_model']:
                self.load_model()

    def configure_data(self, data_x, data_y, batch_creator):
        self._data_x = data_x
        self._data_y = data_y
        self._batch_creator = batch_creator
        self._data_configured = True

    def get_model_path(self):
        if 'model_dir' in self._params:
            model_dir = self._params['model_dir']
        else:
            model_path = './fwd/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if 'model_name' in self._params:
            model_name = self._params['model_name']
        else:
            model_name = 'fwd_model.ckpt'

        return model_dir+model_name


    def load_model(self):
        load_tf_check_point(session=self._sess, filename=self.get_model_path())

    def save_model(self):
        save_path = self._saver.save(self._sess, self.get_model_path())
        print("Model saved in file: %s" % save_path)


    def get_data(self):
        round_complete = None 
        if self._params['batch_params'] is not None:
            if self._batch_creator is not None:
                self._data_x, self._data_y, round_complete = self._batch_creator.get_batch(random_samples=self._params['batch_params']['use_random_batches'])
            else:
                raise Exception("Batch training chosen but batch_creator not configured")

        if self._params['cnn_params'] is None:
            feed_dict = {self._net_ops['x']:self._data_x, self._net_ops['y']:self._data_y}
        else:
            feed_dict = {self._net_ops['image_input']:self._data_x, self._net_ops['y']:self._data_y}
        
        return feed_dict, round_complete

    def train(self, epochs):

        if not self._data_configured:
            raise Exception("Data not configured, please configure..")
        
        if self._params['write_summary']:
            tf.global_variables_initializer().run()
        
        loss = np.zeros(epochs)
        
        feed_dict, _ = self.get_data()

        if self._tf_sumry_wrtr is not None:

            for i in range(epochs):

                print "Starting epoch \t", i
                round_complete = False

                while not round_complete:

                    if self._params['batch_params'] is not None:
                        feed_dict, round_complete = self.get_data()
                    else:
                        #this is to take care of the case when we are not doing batch training.
                        round_complete = True

                    if round_complete:
                        print "Completed round"
    
                    if i % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, loss[i] = self._sess.run(fetches=[self._tf_sumry_wrtr._merged, self._net_ops['train_step']],
                                                    feed_dict=feed_dict,
                                                    options=run_options,
                                                    run_metadata=run_metadata)
                        
                        self._tf_sumry_wrtr.add_run_metadata(metadata=run_metadata, itr=i)
                        self._tf_sumry_wrtr.add_summary(summary=summary, itr=i)
                        print('Adding run metadata for', i)
                    else:  # Record a summary
                        summary, loss[i] = self._sess.run(fetches=[self._tf_sumry_wrtr._merged, self._net_ops['train_step']], 
                                                    feed_dict=feed_dict)
                        self._tf_sumry_wrtr.add_summary(summary=summary, itr=i)
           
            self._tf_sumry_wrtr.close_writer()

        else:
            with tf.device(self._device):
                # Keeping track of loss progress as we train
                train_step = self._net_ops['train_step']
                loss_op  = self._net_ops['cost']

                for i in range(epochs):
                  _, loss[i] = self._sess.run([train_step, loss_op], feed_dict=feed_dict)
  
        return loss

    def run_op(self, op_name, x_input):
        with tf.device(self._device):
            op = self._net_ops[op_name]

            out = self._sess.run(op, feed_dict={self._net_ops['x']: x_input})

            return out
