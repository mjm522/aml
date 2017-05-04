import os
import numpy as np
from aml_io.io_tools import load_data

class LoadPreprocessData():
    
    def __init__(self, file_location=None, filename_prefix=None):
        if file_location is None:
            self._file_location =  os.environ['AML_DATA'] + '/aml_dl/pre_process_data_siamese/'
        else:
            self._file_location = file_location

        if filename_prefix is None:
            self._filename_prefix = 'test_push_data_pre_processed'
        else:
            self._filename_prefix = filename_prefix

        self._file_names = self.get_file_names()
        if not self._file_names:
            raise Exception("File list empty, check the path, or prefix name")

        self._total_files = len(self._file_names)
        self._curr_file_idx =  0
        self._nxt_file_idx  =  (self._curr_file_idx + 1)%self._total_files

    def get_file_names(self):
        data_names = []
        for file in os.listdir(self._file_location):
            if file.startswith(self._filename_prefix):
                data_names.append(self._file_location+file)
        return data_names

    def update_file_idx(self):
        self._curr_file_idx =  self._nxt_file_idx
        self._nxt_file_idx  =  (self._curr_file_idx + 1)%self._total_files

    def load_data(self):
        print "Current batch file index: \t", self._file_names[self._curr_file_idx], "\n"
        data = load_data(self._file_names[self._curr_file_idx])
        self.update_file_idx()
        return data['x'], data['y']

