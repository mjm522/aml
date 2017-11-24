IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 960
PIXELS_PER_METER = 20.

FIN_WIDTH = 0.5
FIN_LENGTH = 2


base_params ={
    'type':'static',
    'pos':(15,20),
    'dim':(4,0.5),
    'color':(127, 127, 127, 255),
}


link1_params = {
        'type':'dynamic',
        'pos':(base_params['pos'][0], base_params['pos'][1]+base_params['dim'][1]/2+FIN_LENGTH),
        'ori':0.,
        'den':1,
        'mu':0.3,
        'dim':(FIN_WIDTH,FIN_LENGTH),
        'color':(255, 127, 127, 255),
        'lin_damp':0.6,
        'ang_damp':0.05,
        'awake':True,
}

link2_params = {
        'type':'dynamic',
        'pos':(link1_params['pos'][0], link1_params['pos'][1]+1.75*FIN_LENGTH),
        'ori':0.,
        'den':1,
        'mu':0.3,
        'dim':(FIN_WIDTH,FIN_LENGTH),
        'color':(127, 255, 127, 255),
        'lin_damp':0.6,
        'ang_damp':0.05,
        'awake':True,
}


link3_params = {
        'type':'dynamic',
        'pos':(link2_params['pos'][0], link2_params['pos'][1]+1.75*FIN_LENGTH),
        'ori':0.,
        'den':1,
        'mu':0.3,
        'dim':(FIN_WIDTH,FIN_LENGTH),
        'color':(127, 127, 255, 255),
        'lin_damp':0.6,
        'ang_damp':0.05,
        'awake':True,
}


joint1_params = {
    'anchor':(link1_params['pos'][0], link1_params['pos'][1]-FIN_LENGTH),
    'lowerAngle':-0.5 ** 3.1415, # -90 degrees
    'upperAngle':0.5 ** 3.1415, #  45 degrees
    'enableLimit':False,
    'maxMotorTorque':10,
    'motorSpeed':0.0,
    'enableMotor':True,
    }

joint2_params = {
    'anchor':(joint1_params['anchor'][0], joint1_params['anchor'][1]+1.75*FIN_LENGTH),
    'lowerAngle':-0.5 ** 3.1415, # -90 degrees
    'upperAngle':0.5 ** 3.1415, #  45 degrees
    'enableLimit':False,
    'maxMotorTorque':0.0,
    'motorSpeed':0.,
    'enableMotor':True,
    }


joint3_params = {
    'anchor':(joint2_params['anchor'][0], joint2_params['anchor'][1]+1.75*FIN_LENGTH),
    'lowerAngle':-0.5 ** 3.1415, # -90 degrees
    'upperAngle':0.5 ** 3.1415, #  45 degrees
    'enableLimit':False,
    'maxMotorTorque':0.0,
    'motorSpeed':0.,
    'enableMotor':True,
    }


man_config = {
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pixels_per_meter': PIXELS_PER_METER,
    'links':[base_params, link1_params,link2_params, link3_params],
    'joints':[joint1_params, joint2_params, joint3_params],
}

box_config ={
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pixels_per_meter': PIXELS_PER_METER,
    'pos':(base_params['pos'][0]-5, base_params['pos'][1]-5),
    'ori':0,
    'dim':(1.,1.),
    'lin_damp':0.6,
    'ang_damp':0.05,
    'awake':True,
    'den':1,
    'mu':0.3,
    'draw_corners':True,
}

hole_config = {
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'pixels_per_meter': PIXELS_PER_METER,
    'type':'static',
    'pos':(base_params['pos'][0]+base_params['dim'][0], base_params['pos'][1]+5),
    'ori':0.,
    'vertices':[(-1.5,-0.5), (0.5,-1), (1,-0.5), (1,1.5), (0.5,1.5), (-0.5,1.3)],
    'color':(125,125,125,255),
    'draw_corners':False,
}


pih_world_config = {

    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'box_config':box_config,
    'man_config':man_config,
    'hole_config':hole_config,
    'no_samples':20,
    'fps': 15,
    'dt': 0.01  ,#0.0167,
    'steps_per_frame': 15,
    'window_caption': 'BoxWorld',
    'pixels_per_meter': PIXELS_PER_METER,
    'push_mag': 0.05,
    'pre_push_offset':0.05,
}