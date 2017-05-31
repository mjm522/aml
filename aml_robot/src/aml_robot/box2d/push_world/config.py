import os

data_folder_path = os.environ.get('AML_DATA')+'/aml_dl/box2d_push_data'

pre_process_folder_path = os.environ.get('AML_DATA')+'/aml_dl/box2d_pre_processed_data/test'

if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)

if not os.path.exists(pre_process_folder_path):
	os.makedirs(pre_process_folder_path)

config = {
	
	'image_width': 640,
	'image_height': 480,
	'pixels_per_meter': 20.0,
	'no_samples':100,
	'fps': 30,
	'dt': 0.0167,
	'window_caption': 'BoxWorld',
	'box_dim': (2,1),
	'record_training_data': True,
	'training_data_file': data_folder_path+'/data_test.pkl',
	'steps_per_frame': 15,
	'data_folder_path':data_folder_path,

}

pre_process_config = {
    'samples_per_file':20,
    'file_name_prefix':'pre_process_box2d_push_data',
    'data_folder_path':pre_process_folder_path,
}