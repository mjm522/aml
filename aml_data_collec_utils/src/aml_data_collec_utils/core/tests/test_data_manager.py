from aml_data_collec_utils.core.data_manager import DataManager

data_man = DataManager(data_name_prefix='test_push_data')

ids=range(0,5)
x_keys = ['box_pos', 'box_ori', 'task_action']
y_keys = ['box_pos', 'box_ori']
x_sub_keys = [[None],[None],['push_xz']]
y_sub_keys = [[None],[None]]

# data_x, data_y = data_man.pack_sample(x_keys=x_keys, x_sub_keys=x_sub_keys, 
#                                           y_keys=y_keys, y_sub_keys=y_sub_keys, 
#                                           ids = ids)

# data_list = data_man.select_data(ids)

# print data_list[0].get(0, x_keys)

data_file_range = range(1,174)

# data_x = data_man.pack_data_x(keys=x_keys, sub_keys=x_sub_keys, ids=ids, just_before=True)
# data_y = data_man.pack_data_y(keys=y_keys, sub_keys=y_sub_keys, ids=ids, just_after=True)

data_x, data_y = data_man.pack_data_in_range_xy(x_keys=x_keys, y_keys=y_keys, x_sub_keys=x_sub_keys, y_sub_keys=y_sub_keys, ids=ids, before_after=True, data_file_range=data_file_range)

print "Data x \n", data_x

print "Data y \n", data_y