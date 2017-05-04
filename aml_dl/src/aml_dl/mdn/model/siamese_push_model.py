import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_io.tf_io import load_tf_check_point
from aml_dl.mdn.model.tf_models import tf_siamese_model
from aml_dl.utilities.tf_summary_writer import TfSummaryWriter


import cv2


class SiamesePushModel(object):
    
    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._device = self._params['device']

        self._tf_sumry_wrtr = None

        self._optimiser = network_params['optimiser']

        self._data_configured = False

        with tf.device(self._device):

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

            self._net_ops = tf_siamese_model(loss_type='quadratic',
                                     cnn_params=network_params['cnn_params'], 
                                     fc_params=network_params['fc_params'],
                                     mdn_params=network_params['inv_params'],
                                     optimiser_params=network_params['optimiser'],
                                     cost_weights=network_params['cost_weights'],
                                     tf_sumry_wrtr=self._tf_sumry_wrtr)

            self._init_op = tf.initialize_all_variables()

            self._saver = tf.train.Saver()

    def init_model(self, epoch = None):
        with tf.device(self._device):
            self._sess.run(self._init_op)

            if self._params['load_saved_model']:
                self.load_model(epoch = epoch)

    def configure_data(self, data_x, data_y, batch_creator):
        if data_x is not None:
            self._data_x_t   = [_x[0] for _x in tmp_x] #every first element of the tuple
            self._data_x_t_1 = [_x[1] for _x in tmp_x] #every second element of the tuple
            data_y_point_len = len(data_y[0])/2 #a single line consists of data at t and t+1
            data_y_array     = np.asarray(data_y)
            data_y_t         = data_y_array[:,0:data_y_point_len]
            data_y_t_1       = data_y_array[:,data_y_point_len:]
            # self._data_y     = data_y_t_1[:, :-self._params['fc_params']['action_dim']].tolist()
            self._action_t   = data_y_t[:, self._params['fc_params']['state_dim']:].tolist()

        else:
            self._data_x_t   = None
            self._data_x_t_1 = None
            self._action_t   = None
            # self._data_y     = None
        
        self._batch_creator = batch_creator
        self._data_configured = True

    def get_model_path(self, subscript=None):

        model_name_subscript = '_'

        if subscript is not None:
            if not isinstance(subscript, str):
                subscript = str(subscript)
            model_name_subscript = subscript + '_'

        if 'model_dir' in self._params:
            model_dir = self._params['model_dir']
        else:
            model_path = './siam/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if 'model_name' in self._params:
            model_name = model_name_subscript + self._params['model_name']
        else:
            model_name = model_name_subscript + 'siam_model.ckpt'

        return model_dir+model_name


    def load_model(self, epoch=None):
        '''
        question: is it better to give filename directly or give epoch number?
        '''
        load_tf_check_point(session=self._sess, filename=self.get_model_path(epoch))

    def save_model(self, epoch=None):
        save_path = self._saver.save(self._sess, self.get_model_path(epoch))
        print("Model saved in file: %s" % save_path)

    def get_data(self):
        round_complete = False 
        if self._params['batch_params'] is not None:
            if self._batch_creator is not None:
                tmp_x, tmp_y, round_complete = self._batch_creator.get_batch(random_samples=self._params['batch_params']['use_random_batches'])
                self._data_x_t   = [_x[0] for _x in tmp_x] #every first element of the tuple
                self._data_x_t_1 = [_x[1] for _x in tmp_x] #every second element of the tuple
                # for x in tmp_x:
                #     cv2.imshow("Before:", np.transpose(np.reshape(x[0],(3,640,480)), axes=[2,1,0]))
                #     cv2.imshow("After:", np.transpose(np.reshape(x[1],(3,640,480)), axes=[2,1,0]))
                #     cv2.waitKey(0)
                data_y_point_len = len(tmp_y[0])/2 #a single line consists of data at t and t+1
                data_y_array     = np.asarray(tmp_y)
                data_y_t         = data_y_array[:,0:data_y_point_len]
                data_y_t_1       = data_y_array[:,data_y_point_len:]
                # self._data_y     = data_y_t_1[:, :-self._params['fc_params']['action_dim']].tolist()
                self._action_t   = data_y_t[:, self._params['fc_params']['state_dim']:].tolist()
            else:
                raise Exception("Batch training chosen but batch_creator not configured")

            feed_dict = {self._net_ops['image_input_t']:self._data_x_t, self._net_ops['image_input_t_1']:self._data_x_t_1, self._net_ops['mdn_y']: self._action_t}
        
        return feed_dict, round_complete

    def train(self, epochs, chk_pnt_save_invl=500):

        if not self._data_configured:
            raise Exception("Data not configured, please configure..")

        with tf.device(self._device):
        
            if self._params['write_summary']:
                tf.global_variables_initializer().run()
            
            loss = np.zeros(epochs)
            
            feed_dict, _ = self.get_data()

            if self._tf_sumry_wrtr is not None:

                for i in range(epochs):

                    print "Starting epoch \t", i
                    round_complete = False
                    batch_no = 0

                    while not round_complete:

                        if self._params['batch_params'] is not None:
                            feed_dict, round_complete = self.get_data()
                        else:
                            #this is to take care of the case when we are not doing batch training.
                            round_complete = True

                        print "Batch number \t", batch_no
                        batch_no += 1
        
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

                        if round_complete:
                            print "That was the last round of epoch %d"%i

                        if i%chk_pnt_save_invl==0 and i!=0:
                            self.save_model(epoch=i)
           
                self._tf_sumry_wrtr.close_writer()

            else:
                with tf.device(self._device):
                    # Keeping track of loss progress as we train
                    train_step = self._net_ops['train_step']
                    loss_op  = self._net_ops['cost']

                    for i in range(epochs):
                        print "Starting epoch \t", i
                        round_complete = False
                        batch_no = 0
                        while not round_complete:
                            if self._params['batch_params'] is not None:
                                feed_dict, round_complete = self.get_data()
                            else:
                                #this is to take care of the case when we are not doing batch training.
                                round_complete = True

                            print "Batch number \t", batch_no
                            batch_no += 1
                            _, loss[i] = self._sess.run([train_step, loss_op], feed_dict=feed_dict)

                            if round_complete:
                                print "That was the last round of epoch %d"%i

                        if i%chk_pnt_save_invl==0 and i!=0:
                            self.save_model(epoch=i)

                    np.savetxt('loss_values.txt', np.asarray(loss))
                    plt.figure()
                    plt.plot(loss)
                    plt.show()

  
        return loss


    def train2(self, iterations, chk_pnt_save_invl=10):

        if not self._data_configured:
            raise Exception("Data not configured, please configure..")

        with tf.device(self._device):
        
            if self._params['write_summary']:
                tf.global_variables_initializer().run()
            
            loss = np.zeros(iterations)
            loss_mdn = np.zeros(iterations)
            loss_fwd = np.zeros(iterations)
            
            feed_dict, _ = self.get_data()

            xs = []
            ys = []

            xs2 = []
            ys2 = []

            xs3 = []
            ys3 = []

            fig1 = plt.figure()
            fig2 = plt.figure()
            fig3 = plt.figure()
            plt.ion()
            plt.show()
            loss_tmp = 0.0
            loss_tmp2 = 0.0
            loss_tmp3 = 0.0

            # Keeping track of loss progress as we train
            train_step = self._net_ops['train_step']
            loss_op  = self._net_ops['cost']
            loss_mdn_op  = self._net_ops['mdn_loss']
            loss_fwd_op  = self._net_ops['fwd_cost']
            for i in range(iterations):
                print "Starting epoch \t", i
                round_complete = False
                if self._params['batch_params'] is not None:
                    feed_dict, round_complete = self.get_data()
                else:
                    #this is to take care of the case when we are not doing batch training.
                    round_complete = True

                _, loss[i], loss_mdn[i], loss_fwd[i] = self._sess.run([train_step, loss_op, loss_mdn_op, loss_fwd_op], feed_dict=feed_dict)
                loss_tmp += loss[i]
                loss_tmp2 += loss_mdn[i]
                loss_tmp3 += loss_fwd[i]
                if round_complete:
                    print "That was the last round of epoch %d"%i
                if i%chk_pnt_save_invl==0 and i!=0:
                    self.save_model(epoch=i)


                if i%10 == 0:
                    print "Iteration %d loss %f"%(i,loss_tmp/10)
                    
                    xs.append(i)
                    ys.append(loss_tmp/10)

                    xs2.append(i)
                    ys2.append(loss_tmp2/10)

                    xs3.append(i)
                    ys3.append(loss_tmp3/10)

                    fig1.add_subplot(111).plot(xs, ys, 'b')
                    fig1.canvas.flush_events()


                    fig2.add_subplot(111).plot(xs2, ys2, 'g')
                    fig2.canvas.flush_events()
                    

                    fig3.add_subplot(111).plot(xs3, ys3, 'r')
                    fig3.canvas.flush_events()
 
                    plt.draw()

                    loss_tmp = loss_tmp2 = loss_tmp3 = 0.0
                
                

                

            
            np.savetxt('loss_values.txt', np.asarray(loss))
                #plt.figure()
                #plt.plot(loss)
                #plt.show()
        return loss

    def test(self, iterations):

        if not self._data_configured:
            raise Exception("Data not configured, please configure..")

        with tf.device(self._device):
        
            if self._params['write_summary']:
                tf.global_variables_initializer().run()
            
            loss = np.zeros(iterations)
            loss_mdn = np.zeros(iterations)
            loss_fwd = np.zeros(iterations)
            
            feed_dict, _ = self.get_data()

            xs = []
            ys = []

            xs2 = []
            ys2 = []

            xs3 = []
            ys3 = []

            fig1 = plt.figure()
            fig2 = plt.figure()
            fig3 = plt.figure()
            plt.ion()
            plt.show()
            loss_tmp = 0.0
            loss_tmp2 = 0.0
            loss_tmp3 = 0.0

            # Keeping track of loss progress as we train
            loss_op  = self._net_ops['cost']
            loss_mdn_op  = self._net_ops['mdn_loss']
            loss_fwd_op  = self._net_ops['fwd_cost']

            round_complete = False
  
            for i in range(iterations):

                if self._params['batch_params'] is not None:
                    feed_dict, round_complete = self.get_data()
                else:
                    #this is to take care of the case when we are not doing batch training.
                    round_complete = True

                loss[i], loss_mdn[i], loss_fwd[i] = self._sess.run([loss_op, loss_mdn_op, loss_fwd_op], feed_dict=feed_dict)

                loss_tmp += loss[i]
                loss_tmp2 += loss_mdn[i]
                loss_tmp3 += loss_fwd[i]


                if round_complete:
                    print "That was the last round of epoch %d"%i



                if i%1 == 0:
                    print "Iteration %d loss %f"%(i,loss_tmp/10)
                    
                    xs.append(i)
                    ys.append(loss_tmp/10)

                    xs2.append(i)
                    ys2.append(loss_tmp2/10)

                    xs3.append(i)
                    ys3.append(loss_tmp3/10)

                    fig1.add_subplot(111).plot(xs, ys, 'b')
                    fig1.canvas.flush_events()


                    fig2.add_subplot(111).plot(xs2, ys2, 'g')
                    fig2.canvas.flush_events()
                    

                    fig3.add_subplot(111).plot(xs3, ys3, 'r')
                    fig3.canvas.flush_events()
 
                    plt.draw()

                    loss_tmp = loss_tmp2 = loss_tmp3 = 0.0
                
                

                

            
            np.savetxt('loss_values.txt', np.asarray(loss))
                #plt.figure()
                #plt.plot(loss)
                #plt.show()
        return loss

    def run_op(self, op_name, image_input_t, image_input_t_1):
        with tf.device(self._device):
            op = self._net_ops[op_name]

            out = self._sess.run(op, feed_dict={self._net_ops['image_input_t']: image_input_t, self._net_ops['image_input_t_1']: image_input_t_1})

            return out


    def run_loss(self, xs , ys_fwd, ys_inv):
        with tf.device(self._device):
            op = self._net_ops['cost']

            feed_dict = {self._net_ops['image_input_t']: xs[:-1], self._net_ops['image_input_t_1']: xs[1:], self._net_ops['mdn_y']: ys_inv[:-1]}

            out = self._sess.run(op, feed_dict=feed_dict)

            return out