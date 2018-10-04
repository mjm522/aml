KEYBOARD_ROBOT_CONFIG = {}



OS_SAWYER_CONFIG = {
    
    'rate': 500, # rate of the controller
    'ctrlr_type':'pos', #other options are 'vel', 'torq'
    'robot_max_speed': 0.20,
    'robot_min_speed': 0.01,
    'ori_speed_ratio': 2.0, # ----- ratio of ori change rate wrt translation
    'custom_controls': None # ----- for custom keyboard commands, provide dict{'key':func}
}

OS_BAXTER_CONFIG = {
    
    'rate': 500, # rate of the controller
    'ctrlr_type':'pos', #other options are 'vel', 'torq'
    'robot_max_speed': 10.0,
    'robot_min_speed': 0.01,
    'ori_speed_ratio': 2.0, # ----- ratio of ori change rate wrt translation
    'custom_controls': None, # ----- for custom keyboard commands, provide dict{'key':func}
}