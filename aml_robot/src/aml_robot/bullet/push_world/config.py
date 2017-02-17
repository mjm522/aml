import os.path
from aml_io.io_tools import get_aml_package_path

model_folder_path  = get_aml_package_path('aml_robot') + '/src/aml_robot/bullet/models/box_push_setup/'
data_storage_path  = get_aml_package_path('aml_data_collec_utils') + '/data/simulation/'


if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)

CAM_INFO ={
	'up':[0.,0.,1.],
	'pos':[1.,1.,1.],
	'target_pos':[0.,0.,0.],
	'distance':4.,
	'yaw':40.,
	'pitch':10.,
	'roll':0.,
	'up_axis_index':2,
	'image_height':1080,
	'image_width':1920,
	'fov':60,
	'near_plane':0.01,
	'far_plane':1000,
}

BOX_TYPE_1 = {
	'length':1.,
	'breadth':1.,
	'height':1.,
}

WORK_SPACE_LIMITS ={
	'x_upper':3.0,
	'x_lower':-3.0,
	'y_upper':3.0,
	'y_lower':-3.0,
	'z_upper':3.0,
	'z_lower':-3.0,
}


config_push_world = {
	'cam':CAM_INFO,
	'image_width': 640,
	'image_height': 480,
	'pixels_per_meter': 20.0,
	'fps': 60,
	'world_path':model_folder_path+'plane.urdf',
	'box_path':model_folder_path+'cube.urdf',
	'robot_path':model_folder_path+'finger.urdf',
	'data_folder_path':data_storage_path,
	'dt': 0.0167,
	'record_training_data': True,
	'training_data_file': 'data_test.pkl',
	'steps_per_frame': 5,
	'camera_pos': [0.0, 0.0, 1.65, 1., -90., 90.],
	'push_force_magnitude':40,
	'reset_spot_offset':[0.,0.,0.],
	'end_effector_tip_offset':[0.,0.,0.],
	'pre_push_offsets':[1.25, 1.25, 0.4],
	'pre_reset_offsets': [0.0, 0.20, 0.0], # w.r.t to box frame
	'reset_spot_offset': [0.0, 0.0, 0.0], # w.r.t to box frame
	'box_center_offset':[0.,0.,0.],
	'box_type':BOX_TYPE_1,
	'work_space_limits':WORK_SPACE_LIMITS,
}