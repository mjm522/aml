import random
import numpy as np
from collections import deque
from multiprocessing import Process
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files


class Buffer():

    def __init__(self, buffer_size):
        self._storage =  deque()
        self._size =  buffer_size
        self._in_use = False

    @property
    def size(self):
        return len(self._storage)

    @property
    def check_overflow(self):
        if len(self._storage) >= self._size:
            return True
        else:
            return False

    @property
    def check_underflow(self):
        if len(self._storage) == 0:
            return True
        else:
            return False

    @property
    def check_in_use(self):
        return self._in_use

    def add_to_buffer(self, data):

        if not self.check_in_use and not self.check_overflow:
            self._storage.append(data)
            return True
        else:
            return False


class MakeBatch():

    def __init__(self, batch_size):
        self._size = batch_size
        self._buffer = None
        self._is_configured = False

    def configure_buffer(self, new_buffer):
        self._buffer = new_buffer
        self._buffer._in_use = True
        self._is_configured = True

    def get_mini_batch(self, random_samples=False):

        if self._buffer is None:
            raise Exception("No buffer configured, quitting")

        if self._buffer.check_underflow:
            raise Exception("Empty buffer configured, quitting")

        if random_samples:
            batch = random.sample(self._buffer._storage, self._size)
        else:
            if self._buffer:
                batch = [self._buffer._storage.popleft() for _ in range(self._size)]
            else:
                self._buffer._full = False
                self._buffer._in_use = False
                print "Configured buffer is empty"
                return None, None
        #x
        x_batch = [item[0] for item in batch]
        x_batch = np.array(x_batch)
        #y       
        y_batch = [item[1] for item in batch]
        y_batch = np.array(y_batch)

        return x_batch, y_batch


class BatchCreator(DataManager):
    def __init__(self, batch_params):

        if batch_params['buffer_size']%batch_params['batch_size'] !=0:
            print "WARNING: It is better to have a buffer size that is a multiple of batch size"

        self._buffer_1 = Buffer(batch_params['buffer_size'])
        self._buffer_2 = Buffer(batch_params['buffer_size'])
        self._batch = MakeBatch(batch_params['batch_size'])
        self._parmams = batch_params
        
        #THIS is in case we want to make it multi threaded
        # self._ready = False

        # p1 = Process(name='loading to buffer',  target=self.load_data_to_buffer)
        # p2 = Process(name='configuring buffer', target=self.configure_buffer)
        # p = Process(name='run in BatchCreator', target=self.run)

        # p1.daemon = True #blocking run
        # p2.daemon = False #not blocking run
        # p.daemon = False #not blocking main thread
        
        # p1.start()
        # p2.start()
        # p.start()

        self.load_data_to_buffer()
        self.configure_buffer()


    def get_batch(self, random_samples=False):
        # if self._ready:
        return self._batch.get_mini_batch(random_samples)
        # else:
        #     print "WARNING: Not yet ready, buffers filling up"
        #     return None, None

    def load_data_to_buffer(self):

        data_x, data_y = get_data_from_files(data_file_range=self._parmams['data_file_indices'], 
                                             model_type=self._parmams['model_type'])
        
        for x,y in zip(data_x,data_y):
            if (self._buffer_1.add_to_buffer((x,y))):
                continue
            else:
                self._buffer_2.add_to_buffer((x,y))

    def configure_buffer(self):
    
        if self._buffer_1.check_overflow and not self._buffer_2.check_in_use and not self._buffer_1.check_underflow:
            self._batch.configure_buffer(self._buffer_1)
            self._buffer_1._in_use = True
            self._buffer_2._in_use = False
        elif self._buffer_2.check_overflow and not self._buffer_1.check_in_use and not self._buffer_2.check_underflow:
            self._batch.configure_buffer(self._buffer_2)
            self._buffer_2._in_use = True
            self._buffer_1._in_use = False
