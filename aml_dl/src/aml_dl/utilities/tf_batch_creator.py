import random
import numpy as np
from aml_io.convert_tools import string2image
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files
from aml_dl.mdn.utilities.get_pre_process_data import LoadPreprocessData

class BatchCreator(DataManager):
    def __init__(self, batch_params):
        self._params = batch_params
        self._size      =  self._params['batch_size']
        self._x_buffer  =  None
        self._y_buffer  = None
        self._start_idx = 0
        self._end_idx   = 0
        self._round_complete = False
        self._buffer_size = None
        self._finish_reading_files = False

        self._data_file_range = self._params['data_file_indices']
        self._files_per_read  = self._params['files_per_read']
        self._file_idx_start  = self._data_file_range[0]
        self._load_pre_processd_data = self._params['load_pre_processd_data']

        if self._load_pre_processd_data:
            
            if 'filename_prefix' in self._params.keys():
                filename_prefix = self._params['filename_prefix']
            else:
                filename_prefix = None
            
            self._get_pre_processd_data = LoadPreprocessData(file_location=self._params['data_dir'],
                                                             filename_prefix=filename_prefix,)
    
        self.start_batcher()

    def start_batcher(self):
        self._file_idx_end  = self._file_idx_start + self._files_per_read
        if self._file_idx_end > self._data_file_range[-1]:
            self._finish_reading_files = True
            self._file_idx_end = self._data_file_range[-1]
        
        self.load_data_to_buffer(range(self._file_idx_start, self._file_idx_end))
        
        if self._finish_reading_files:
            self._file_idx_start = 0
            self._finish_reading_files = False
        else:
            self._file_idx_start = self._file_idx_end

    def load_data_to_buffer(self, data_file_range):

        if not self._load_pre_processd_data:

            tmp_x, self._y_buffer = get_data_from_files(data_file_range=data_file_range, 
                                                        model_type=self._params['model_type'])

            #if input is an image, then we need to convert the string to float
            if self._params['model_type'] == 'cnn' or self._params['model_type'] == 'siam':
                self._x_buffer = []
                for x_image in tmp_x:
                    if self._params['model_type'] == 'siam':
                        self._x_buffer.append((np.transpose(string2image(x_image[0][0]), axes=[2,1,0]).flatten(), 
                                               np.transpose(string2image(x_image[1][0]), axes=[2,1,0]).flatten()))
                    else:
                        self._x_buffer.append(np.transpose(string2image(x_image[0]), axes=[2,1,0]).flatten())
            else:
                self._x_buffer = tmp_x

        else:
            self._x_buffer, self._y_buffer = self._get_pre_processd_data.load_data()

        assert(len(self._x_buffer) == len(self._y_buffer))
        self._buffer_size = len(self._x_buffer)


    def get_batch(self, random_samples=False):

        self._start_idx = self._end_idx
        self._end_idx  += self._size

        if self._end_idx > self._buffer_size:
            rest = self._size - (self._buffer_size - self._start_idx)
            indices = range(self._start_idx, self._buffer_size) + range(0, rest)
            self._end_idx = self._size
            self._start_idx = 0
            self._round_complete = True
        else:
            indices = range(self._start_idx, self._end_idx)
            self._round_complete = False
            
        if random_samples:
            indices = [random.randint(0, self._buffer_size) for _ in range(self._size)]
            
        x_batch = [self._x_buffer[idx] for idx in indices]
        y_batch = [self._y_buffer[idx] for idx in indices]

        if self._round_complete:
            self.start_batcher()

        return x_batch, y_batch, self._round_complete

