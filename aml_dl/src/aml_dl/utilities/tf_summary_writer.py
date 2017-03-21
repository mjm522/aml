import os
import numpy as np
import tensorflow as tf


class TfSummaryWriter():

    def __init__(self, tf_session, summary_dir=None):

        self._sess = tf_session
        
        if summary_dir is None:
            self._summary_dir = './summary'
        else:
            self._summary_dir = summary_dir

        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        if tf.gfile.Exists(self._summary_dir):
            tf.gfile.DeleteRecursively(self._summary_dir)

        self._summary = tf.summary

    def write_summary(self):
        self._merged = self._summary.merge_all()
        train_summary = self._summary_dir+'/train'
        test_summary = self._summary_dir+'/test'
        
        if not os.path.exists(train_summary):
            os.makedirs(train_summary)

        if not os.path.exists(test_summary):
            os.makedirs(test_summary)

        self._train_writer = self._summary.FileWriter(train_summary, self._sess.graph)
        self._test_writer = self._summary.FileWriter(test_summary)        

    def add_variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            self.add_scalar(name_scope='mean', data=mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            self.add_scalar(name_scope='stddev', data=stddev)
            self.add_scalar(name_scope='max', data=tf.reduce_max(var))
            self.add_scalar(name_scope='min', data=tf.reduce_min(var))
            self.add_histogram(name_scope='histogram', data=var)


    def add_histogram(self, name_scope, data):
        self._summary.histogram(name_scope, data)

    def add_scalar(self, name_scope, data):
        self._summary.scalar(name_scope, data)

    def add_image(self, name_scope, image, size=10):
        self._summary.image(name_scope, image, size)

    def add_summary(self, summary, itr, op='test'):
        
        if op == 'test':
            self._test_writer.add_summary(summary, itr)
        elif op =='train':
            self._train_writer.add_summary(summary, itr)

    def add_run_metadata(self, metadata, itr, op='test'):

        if op == 'test':
            self._test_writer.add_run_metadata(metadata, 'step%03d' % itr)
        elif op =='train':
            self._train_writer.add_run_metadata(metadata, 'step%03d' % itr)

    def close_writer(self):
        self._train_writer.close()
        self._test_writer.close()
