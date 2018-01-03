import numpy as np
import tensorflow as tf

class TfFunction(object):

    def __init__(self, inputs, outputs, astype=np.float64):
        self._inputs   = inputs
        self._outputs  = outputs
        self._astype   = astype

    def __call__(self, sess, params):
        feeds = {}

        for inp in self._inputs:
            tensor_name = inp.name[:-2]
            if tensor_name in params:
                if params[tensor_name].ndim == 1:
                    data = params[tensor_name][:,None]
                else:
                    data = params[tensor_name]

                feeds[inp] = data

        output = sess.run(self._outputs, feeds)

        if isinstance(output, list):
            output = [out.astype(self._astype) for out in output]
        
        return output
