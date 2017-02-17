
from aml_io.data_manager import DataManager
from aml_io.io_tools import get_aml_package_path

training_data_path = get_aml_package_path('aml_data_collec_utils') + '/data/'

data_manager = DataManager.from_file(training_data_path,['push_data_01.pkl'])

ids=[0,1,2,3,4]

x_keys = ['state_before','task_before']

y_keys = ['state_after','task_after']

x_sub_keys = [['position', 'velocity'],['pos', 'ori']]

y_sub_keys = [['position', 'velocity'],['pos', 'ori']]

# data_x = data_manager.pack_data(keys=x_keys, sub_keys=x_sub_keys, ids=ids)

# data_y = data_manager.pack_data(keys=y_keys, sub_keys=y_sub_keys, ids=ids)

data_x, data_y = data_manager.pack_sample(x_keys=x_keys, x_sub_keys=x_sub_keys, 
	                                      y_keys=y_keys, y_sub_keys=y_sub_keys, 
	                                      ids = ids)

print "X DATA \n", data_x

print "Y DATA \n", data_y

