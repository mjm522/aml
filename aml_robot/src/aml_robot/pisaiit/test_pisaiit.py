#!/usr/bin/env python

import rospy

from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.log_utils import aml_logging

from functools import partial

import numpy as np

logger = aml_logging.get_logger(__name__)

def callback(agent, msg):
    global logger
    # logger.info("Look: %s Count: %d"%(msg,agent.c))



class SomeObj:
    def __init__(self):
        self.c = 0




rospy.init_node('pisa_iit_soft_hand_test', anonymous=True)

obj = SomeObj()

robot = PisaIITHand('right', partial(callback, obj))

cmds_close = np.arange(0.0,1.0,0.05)
cmds_open = cmds_close[::-1]
cmds = np.hstack([cmds_close, cmds_open])
cmd_idx = 0
logger.info(cmds)


rate = rospy.Rate(10)  # 10hz
while not rospy.is_shutdown():
    obj.c += 1

    c = raw_input("Commands (c/o/fc/fo): ")

    try:
        cmds = cmds_close
        cmd = cmds[cmd_idx:cmd_idx+1]

        if c == 'c':
            cmd_idx = np.minimum((cmd_idx+1),len(cmds))
            robot.exec_position_cmd(cmd)
        elif c == 'o':
            cmd_idx = np.maximum((cmd_idx-1),0)
            robot.exec_position_cmd(cmd)
        elif c == 'fc':

            for cm in cmds_close:
                robot.exec_position_cmd(cm)

        elif c == 'fo':

            for cm in cmds_open:
                robot.exec_position_cmd(cm)


        # logger.info("CMD: %lf State %s"%(cmd, robot.state()))
        
    except:
        pass
    # if cmd_idx == (len(cmds)-1):
    #     logger.debug("Exiting demo!")
    #     break

    rate.sleep()
