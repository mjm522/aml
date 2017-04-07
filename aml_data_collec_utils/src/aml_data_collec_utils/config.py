import os
from aml_io.io_tools import get_aml_package_path

data_storage_path = os.environ['AML_DATA'] + '/aml_dl/baxter_push_data/'
print data_storage_path

if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)

#all measurements in meters
#the white box
BOX_TYPE_1 = {
    'length': 0.210, # along x axis
    'breadth':0.153, # along z axis
    'height':0.080,  # along y axis
}

#amazon box
BOX_TYPE_2 = {
    'length': 0.44,
    'breadth':0.34,
    'height':0.27,
}

#cubic box
BOX_TYPE_3 = {
    'length': 0.124,
    'breadth':0.124,
    'height':0.091,
}

#the white table
TABLE = {
    'length': 1.795,
    'breadth':0.84,
    'height':0.72,
}

config_white_rectangular_box = {
    #-0.11549958
    'reset_spot_offset': [-0.06049958, -0.0023311, -0.00379943], # w.r.t to box frame
    'box_center_offset': [-0.055,0.0,0.0],
    'end_effector_tip_offset': [0.0, 0.16090282, 0.0], # w.r.t to box #[0.0, 0.16090282, 0.0]
    'pre_push_offsets': [0.25, 0.12, 0.25], # w.r.t to box frame before: 0.096
    'pre_reset_offsets': [0.0, 0.20, 0.0], # w.r.t to box frame
    'box_type': BOX_TYPE_1,
}

config_cubic_box = {
    #-0.11549958
    'reset_spot_offset': [0.0, 0.0, 0.0], # w.r.t to box frame
    'box_center_offset': [0.0,0.0,0.0],
    'end_effector_tip_offset': [0.0, 0.16090282, 0.0], # w.r.t to box #[0.0, 0.16090282, 0.0]
    'pre_push_offsets': [0.15, 0.12, 0.15], # w.r.t to box frame before: 0.096
    'pre_reset_offsets': [0.0, 0.20, 0.0], # w.r.t to box frame
    'box_type': BOX_TYPE_3,
    'scale_adjust': 0.5,
}


config = {
    'data_folder_path': data_storage_path,
}

config.update(config_cubic_box)
