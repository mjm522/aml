import os

CAM_INFO ={
	'up':[0.,0.,1.],
	'pos':[1.,1.,1.],
	'target_pos':[0.,0.,0.],
	'distance':4.,
	'yaw':40.,
	'pitch':10.,
	'roll':0.,
	'up_axis_index':2,
	'image_height':480,
	'image_width':640,
	'fov':60,
	'near_plane':0.01,
	'far_plane':1000,
}


config = {'cam': CAM_INFO }


SAWYER_BULLET_CONFIG = {
	'ee_link_idx': 16,
	'ee_link_name': 'right_hand'
}


# resources_folder_path = os.environ['GLOVE_TOOLS_RESOURCES']

config_pisa_hand_right = {
	'thumb_joints': [ 'soft_hand_thumb_%s_joint'%(jm,) for jm in ['abd','inner','outer']],
	'index_joints': [ 'soft_hand_index_%s_joint'%(jm,) for jm in ['abd','inner','middle','outer']],
	'middle_joints': [ 'soft_hand_middle_%s_joint'%(jm,) for jm in ['abd','inner','middle','outer']],
	'ring_joints': [ 'soft_hand_ring_%s_joint'%(jm,) for jm in ['abd','inner','middle','outer']],
	'little_joints': [ 'soft_hand_little_%s_joint'%(jm,) for jm in ['abd','inner','middle','outer']],
	"finger_order": ["thumb", "index", "middle","ring", "little"],
	'force_joint_ids': range(10,39,7), # pisaiit finger tips joint ids
	'force_scale': 0.01,
	'force_vis_threshold': 0.05, # if the ratio between max/min forces go above this threshold, then we show the force vector
	'show_debug_info': True,
	'orientation_offset': [3.141592654, 1.570796327, 3.141592654], #3.141592654
	'update_ori': False,
	'update_pos': True,
	'dt': 0.0167,
	'record_training_data': True,
	'training_data_file': 'data_test.pkl',
	'steps_per_frame': 5,
	'ee_link_idx': 16,
	'ee_link_name': 'right_hand'
}


config_hand_world = config_pisa_hand_right