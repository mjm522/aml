
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

#the white table
TABLE = {
    'length': 1.795,
    'breadth':0.84,
    'height':0.72,
}

config = {
    'reset_spot_offset': [-0.10,0,0],
    'end_effector_tip_offset': [0.0, 0.0, 0.16090282],
    'pre_push_offsets': [0.20, 0.09, 0.20],
    'box_type': BOX_TYPE_1,
}