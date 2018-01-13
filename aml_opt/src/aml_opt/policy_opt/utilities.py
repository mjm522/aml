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


def get_feature(observation, reward):
    return np.r_[observation, observation**2, reward, reward**2, reward**3, 1]

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret