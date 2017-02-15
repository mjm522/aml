
from os.path import dirname, abspath
model_folder_path = dirname(dirname(abspath(__file__))) + '/models/'


BOX_TYPE_1 = {
	'length':0.3,
	'breadth':0.3,
	'height':0.1,
}


config_push_world = {
	
	'image_width': 640,
	'image_height': 480,
	'pixels_per_meter': 20.0,
	'fps': 60,
	'box_dim':(0.15, 0.15, 0.15),
	'model_name':model_folder_path+'/box_poke_setup.xml',
	'dt': 0.0167,
	'window_caption': 'MujocoWorld',
	'record_training_data': True,
	'training_data_file': 'data_test.pkl',
	'steps_per_frame': 5,
	'camera_pos': [0.0, 0.0, 1.65, 1., -90., 90.],
	'reset_spot_offset':[0.,0.,0.],
	'end_effector_tip_offset':[0.,0.,0.],
	'pre_push_offsets':[0.05, 0.05, 0.],
	'pre_reset_offsets': [0.0, 0.20, 0.0], # w.r.t to box frame
	'reset_spot_offset': [0.0, 0.0, 0.0], # w.r.t to box frame
	'box_center_offset':[0.,0.,0.],
	'box_type':BOX_TYPE_1
}