import matplotlib.image as mpimg

from aml_io.io_tools import load_data, save_data

import numpy as np

class DataManager(object):

	def __init__(self, data = []):

		self._data = data

	@classmethod
	def from_file(cls,filename):

		return cls(load_data(filename))


	def add(self,sampled_data):

		self._data.append(sampled_data)


	def get_sample(self,idx,key):
		assert( idx < len(self._data) )

		return self._data[idx][key]

	def pack_data_x(self,keys):

		data_x = []

		for datum in self._data:
			x = []
			for key in keys:
				if key in ['state_start','state_end']:
					x = np.r_[x,datum[key]['linear_velocity']]
					# datum[key]['position'], .., datum[key]['angle'],datum[key]['angular_velocity']

			data_x.append(x)


		return data_x

	def pack_data_y(self):

		data_y = []

		for datum in self._data:

			data_y.append([datum['push_action'][0][4]])


		return data_y














