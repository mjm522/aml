KEYBOARD_ROBOT_CONFIG = {}

JS_TELEOP_CTRL = {
    
    'robot_joints':None, #these joints will be one to one mapped
    'haptic_joints':None,
    'scale_from_home': True,
    'robot_home':None, # home position of baxter
    'rate': 500, # rate of the controller
    'ctrlr_type':'pos', #other options are 'vel', 'torq'
}


OS_TELEOP_CTRL = {
    
    'rate': 500, # rate of the controller
    'ctrlr_type':'pos', #other options are 'vel', 'torq'
}