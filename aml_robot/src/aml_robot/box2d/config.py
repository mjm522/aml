import os

data_folder_path = os.environ.get('AML_DATA')+'/aml_dl/box2d_push_data'

if not os.path.exists(data_folder_path):
	os.mkdir(data_folder_path)

config = {
	
	'image_width': 640,
	'image_height': 480,
	'pixels_per_meter': 20.0,
	'fps': 60,
	'dt': 0.0167,
	'window_caption': 'BoxWorld',
	'box_dim': (2,1),
	'record_training_data': True,
	'training_data_file': data_folder_path+'/data_test.pkl',
	'steps_per_frame': 5,
	'data_folder_path':data_folder_path,

}