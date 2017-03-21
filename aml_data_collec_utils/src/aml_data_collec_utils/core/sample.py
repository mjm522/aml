from aml_io.io_tools import load_data, save_data
from aml_io.log_utils import aml_logging

import numpy as np
import pickle

import itertools


class Sample(object):


    next_id = itertools.count().next

    @classmethod
    def configure_id_counter(cls, start_count):
        next_id = itertools.count(start_count).next

    def __init__(self):

        self._contents                = []

        self._is_valid = True

        self._id    = Sample.next_id()


    """
    data: python dictionary of key, value pairs
    """
    def add(self,data):

        self._contents.append(data)

        s = self.size
        aml_logging.info("New data point %d"%(s,))

    def get_contents(self):

        return self._contents

    @property
    def size(self):
        return len(self._contents)

    """
    idx: index for data
    keys: keys in data
    """
    def get(self, idx, keys=None):

        assert( idx < len(self._contents) )

        data = self._contents[idx]

        if keys is None:
            keys = data.keys()
            # TODO add key order in config file
            # adjust key order

        #out = [ data[k] for k in list(set(keys) & set(data)) ]
        out = []

        try:
            out = [data[k] for k in keys]
        except Exception as e:
            print "No such key ", e, " ", keys

        return out

    def set(self, idx, key, value):

        assert( idx < len(self._contents) and len(self._contents) > 0 )

        data = self._contents[idx]

        assert( key in data.keys() )

        data[key] = value

        return data[key]


    def get_keys(self):

        assert( len(self._contents) > 0 )


        return self._contents[0].keys()


    def set_valid(self, valid):
        self._is_valid = valid

    def is_valid(self):
        return self._is_valid


    def get_id(self):
        return self._id


    # for pickling and unpickling
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d