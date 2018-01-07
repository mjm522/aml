import os

IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 960
PIXELS_PER_METER = 180.


#for recording demonstrations
demo_storage_path  = os.environ['AML_DATA'] + '/aml_playground/pih_worlds/box2d_demos/'
if not os.path.exists(demo_storage_path):
    os.makedirs(demo_storage_path)

box_config = {
    
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pixels_per_meter': PIXELS_PER_METER,
    'pos':(1., 1.),
    'ori':0.,
    'window_caption': 'BoxWorld',
    'box_dim': (0.125,0.125),
    'mass':0.100,
    'inertia':0.100,

}

box_fin_config = {
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pos':(2., 2.),
    'ori':0.,
    'dim':(0.05,0.05),
    'f_mag':10.,
    'color':(255,125,125),
    'pixels_per_meter': PIXELS_PER_METER,
}

circle_fin_config = {
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pos':(2., 2.),
    'ori':0.,
    'dim':0.05,
    'f_mag':10.,
    'color':(255,125,125),
    'pixels_per_meter': PIXELS_PER_METER,
}


push_world_config = {

    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'box_config':box_config,
    'fin_config':circle_fin_config,
    'num_fins':2,
    'no_samples':20,
    'fps': 15,
    'dt': 0.01  ,#0.0167,
    'steps_per_frame': 15,
    'window_caption': 'BoxWorld',
    'pixels_per_meter': PIXELS_PER_METER,
    'push_mag': 0.05,
    'pre_push_offset':0.05,
    'demo_storage_path':demo_storage_path,
    'save_demo':True,
}
