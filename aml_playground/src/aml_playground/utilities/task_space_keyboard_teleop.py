

import numpy as np




class KeyboardTeleop:

    def __init__(self, env = None):

        if env is not None:
            self._env = env
        else:
            from aml_rl_envs.sawyer.sawyer_env import SawyerEnv
            self._env = SawyerEnv()

        try:
            self._robot = self._env._robot
        except (NameError,AttributeError):
            print "Couldn't find var  _robot ... Looking for var _sawyer ..."
            self._robot = self._env._sawyer

        self._speed = 0.01
        self._ori_speed_ratio = 0.1

        self._create_bindings()
        self._getch = Getch()


    def _create_bindings(self):


        self._bindings = {
                            # ----- translation
                            'w': (self._move_ee, ['x', 1.0], "move along X"),
                            's': (self._move_ee, ['x', - 1.0], "move along (-X)"),
                            'q': (self._move_ee, ['z', 1.0], "move along Z"),
                            'e': (self._move_ee, ['z', - 1.0], "move along (-Z)"),
                            'a': (self._move_ee, ['y', 1.0], "move along Y"),
                            'd': (self._move_ee, ['y', - 1.0], "move along (-Y)"),

                            # ----- rotation
                            '6': (self._turn_ee, ['x', 1.0], "turn about X"),
                            '4': (self._turn_ee, ['x', - 1.0], "turn about (-X)"),
                            '7': (self._turn_ee, ['z', 1.0], "turn about Z"),
                            '9': (self._turn_ee, ['z', - 1.0], "turn about (-Z)"),
                            '8': (self._turn_ee, ['y', 1.0], "turn about Y"),
                            '5': (self._turn_ee, ['y', - 1.0], "turn about (-Y)"),

                            # ----- speed scale
                            '+': (self._change_speed, [0.01], "increase spead by 0.01"),
                            '-': (self._change_speed, [-0.01], "decrease spead by 0.01"),
                         }


    def _change_speed(self, inc):

        if 0.0 < self._speed < 1.0:
            self._speed += inc
            print "New Speed: %f"%self._speed
        else:
            print "Allowed Speed Limit reached!"

    def _move_ee(self, axis, speed):

        curr_ee_pos = self._robot.state()['ee_point'],
        print curr_ee_pos

        speed*= self._speed

        if axis == 'y':

            pos_des = curr_ee_pos + np.asarray([0,speed,0])

        elif axis == 'z':

            pos_des = curr_ee_pos + np.asarray([0,0,speed])

        elif axis == 'x':

            pos_des = curr_ee_pos + np.asarray([speed,0,0])

        cmd = self._robot.inv_kin(ee_pos = pos_des, ee_ori = None)#, ori_type = 'eul')
        self._robot.apply_action(cmd.tolist(), Kp=None, Kd=None)
        # self._robot.set_joint_state(cmd)
        # print new_pos, new_ori

    def _turn_ee(self, axis, speed):

        curr_ee_pos, curr_ee_ori = self._robot.state()['ee_point'], self._robot.state(ori_type='eul')['ee_ori']
        # print curr_ee_pos, curr_ee_ori
        print self._robot.state()['ee_ori']

        # print self._robot.inv_kin(curr_ee_pos, curr_ee_ori)
        speed*= (self._speed*self._ori_speed_ratio)

        if axis == 'y':

            ori_des = curr_ee_ori + np.asarray([0,speed,0])

        elif axis == 'z':

            ori_des = curr_ee_ori + np.asarray([0,0,speed])

        elif axis == 'x':

            ori_des = curr_ee_ori + np.asarray([speed,0,0])

        # print 'des ori', ori_des

        cmd = self._robot.inv_kin(ee_pos = curr_ee_pos, ee_ori = ori_des, ori_type = 'eul')

        # print cmd, 'cmd'
        self._robot.apply_action(cmd.tolist(), Kp=None, Kd=None)
        # print new_pos, new_ori


    def _teleoperate(self):

        c = self._getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                return True
                
            elif c in self._bindings:
                cmd = self._bindings[c]
                print("command: %s" % (cmd[2],))
                cmd[0](*cmd[1])

            elif c == '/' or '?':
                self._print_help()

            return False


    def run(self):

        done = False
        print("Controlling End Effector. Press ? for help, Esc to quit.")

        while not done:
            done = self._teleoperate()
            self._env.simple_step()

        print "\nExiting ... \n"
                    

    def _print_help(self):
        print("key bindings: ")
        print("  Esc: Quit")
        print("  ?: Help")
        for key, val in sorted(self._bindings.items(),
                                           key=lambda x: x[1][2]):
                        print("  %s: %s" % (key, val[2]))


class Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


if __name__ == '__main__':
    
    from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
    from aml_playground.var_imp_reps.exp_params.experiment_var_imp_params import exp_params
    import copy

    config = copy.deepcopy(exp_params['env_params'])
    env = SawyerEnv(config=config)

    teleop = KeyboardTeleop(env)

    teleop.run()


    # import time
    # while True:
    #     # teleop._move_ee('z',0.001)

    #     pos = env._sawyer.state()['ee_point']
    #     ori = env._sawyer.state()['ee_point']

    #     new_pos = pos + np.asarray([0.00,0.,0.01])
    #     print pos

    #     cmd = env._sawyer.inv_kin(new_pos.tolist())
    #     print 'cmd', new_pos


    #     raw_input()



    #     env._sawyer.set_joint_state(cmd)


    #     env.simple_step()
    #     # time.sleep(0.1)