import os
import numpy as np
import matplotlib.pyplot as plt
from config import box2d_dmp_config
from aml_io.io_tools import save_data, load_data
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_playground.peg_in_hole.pih_worlds.box2d.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.box2d.config import pih_world_config



class Box2dDMP():
    def __init__(self, config):
        self._config    = config
        self._pih_world = PIHWorld(pih_world_config)
        self._pih_manipulator = self._pih_world._manipulator
        self._dmp = DiscreteDMP(config=config)
        self._viewer    = Box2DViewer(self._pih_world, pih_world_config, is_thread_loop=False)



    def view_traj(self, trajectory):

        trajectory *= self._viewer._config['pixels_per_meter']

        trajectory[:, 0] -= self._viewer._config['cam_pos'][0]

        trajectory[:,1] = self._viewer._config['image_height'] - self._viewer._config['cam_pos'][1] - trajectory[:,1]

        self._viewer._demo_point_list = trajectory.astype(int)


    def train_dmp(self):

        self._dmp.load_demo_trajectory(self._demo)
        self._dmp.train()

    def load_demo(self, trajectory):
        self._demo = trajectory

    def test_dmp(self, speed=1., plot_traj=False):

        test_config = self._config
        test_config['dt'] = 0.001

        # play with the parameters
        start_offset = np.array([0.,0.])
        goal_offset = np.array([.0, 0.])
        external_force = np.array([0.,0.,0.,0.])
        alpha_phaseStop = 20.

        test_config['y0'] = self._dmp._traj_data[0, 1:] + start_offset
        test_config['dy'] = np.array([0., 0.])
        test_config['goals'] = self._dmp._traj_data[-1, 1:] + goal_offset
        test_config['tau'] = 1./speed
        test_config['ac'] = alpha_phaseStop
        test_config['type'] = 1

        if test_config['type'] == 3:
            test_config['extForce'] = external_force
        else:
            test_config['extForce'] = np.array([0,0,0,0])

        test_traj = self._dmp.generate_trajectory(config=test_config)['pos']

        if plot_traj:

            plt.figure('x vs y demo_traj & test_traj')
            demo_plt, = plt.plot(self._dmp._traj_data[:,1], self._dmp._traj_data[:,2], 'b-', label='demo_traj')
            traj_plt, = plt.plot(test_traj[:,1], test_traj[:,2], 'r--', label='test_traj')
            plt.legend(handles=[demo_plt, traj_plt])

            plt.figure('x-y vs time of test_traj')
            x_traj_plt, = plt.plot(test_traj[:,0], test_traj[:,1], 'g-', label='x')
            y_traj_plt, = plt.plot(test_traj[:,0], test_traj[:,2], 'm-', label='y')
            plt.legend(handles=[x_traj_plt, y_traj_plt])
            plt.show()


        vel_traj =  np.diff(test_traj[:,1:], axis=0)
        vel_traj =  np.vstack([np.zeros_like(vel_traj[0]), vel_traj])*test_config['dt']
        acc_traj =  np.diff(vel_traj, axis=0)
        acc_traj =  np.vstack([np.zeros_like(acc_traj[0]), acc_traj])*test_config['dt']
        
        test_result = {
        'pos_traj': test_traj[:,1:],
        'vel_traj':vel_traj,
        'acc_traj':acc_traj
        }
        return test_result
    

    def run(self):
        
        self.train_dmp()
        test_result = self.test_dmp(speed=1., plot_traj=False)
        des_path = test_result['pos_traj']

        indices = np.arange(0, len(des_path), 23)

        des_path = des_path[indices, :]

        k = -1
        
        task_complete = False

        manipulator_data = []

        self.view_traj(des_path.copy())

        while self._viewer._running and not task_complete:

            k += 1
            error = 100.

            while error > 0.25:

                set_point = np.hstack([des_path[k,:], 0.05])#np.hstack([self._dmp._traj_data[k,1:], 0.])#np.hstack([des_path[k,:], 0.1]) #np.array([2., 5., -np.pi/2]) #

                data = self._pih_manipulator.get_state()
                data['set_point'] = set_point

                print len(manipulator_data)

                # hack to collect data
                if len(manipulator_data) > 720:
                    task_complete = True
                    break

                action = self._pih_manipulator.compute_os_ctrlr_cmd(os_set_point=set_point, Kp=20)

                self._pih_world.update(action)

                for i in range(self._viewer._steps_per_frame): 
                    self._pih_world.step()

                self._viewer.draw()

                if k == des_path.shape[0]:
                    task_complete = True

                manipulator_ee_pos = self._pih_world.get_state()['manipulator']['ee_pos']
                error = np.linalg.norm(set_point - manipulator_ee_pos)

                data['set_point_index'] = k
                data['ee_error'] = error
                data['action']   = action


                print "Set point \t", np.round(set_point, 3)
                print "EE position \t", np.round(manipulator_ee_pos, 3)
                print "Computed cmd \t", np.round(action, 3)
                print "Error \t", error

                manipulator_data.append(data)

        save_data(manipulator_data, os.environ['AML_DATA'] + '/aml_playground/pih_worlds/box2d/man_data.pkl')


def main():

    data_storage_path = os.environ['AML_DATA'] + '/aml_playground/pih_worlds/box2d/demos/' 
    path_to_demo = data_storage_path + 'demo.pkl'

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    trajectory = np.asarray(load_data(path_to_demo))

    dmp_box2d = Box2dDMP(box2d_dmp_config)
    dmp_box2d.load_demo(trajectory)
    dmp_box2d.run()


if __name__ == '__main__':
    main()