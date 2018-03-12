import os
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from mpl_toolkits.mplot3d import Axes3D
from aml_io.io_tools import save_data,load_data
from aml_rl_envs.hand.hand_obst_env import HandObstacleEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data


class AnalyzeManData():

    def __init__(self, env=None):

        if env is None:
            # self._env = HandObjEnv(action_dim=18, config=HAND_OBJ_CONFIG, randomize_box_ori=False, keep_obj_fixed=True)
            self._env = HandObstacleEnv(action_dim=18, config=HAND_OBJ_CONFIG, randomize_box_ori=False,  keep_obj_fixed=False)
        else:
            self._env = env

        self._data_root = os.environ['AML_DATA'] + '/aml_playground/manipulation/'
        self._data_file_name = self._data_root + 'demo1/collect_man_data.pkl'

        self._sp_obj_file_name = self._data_root + '/surface_point_obj.csv'
        self._sn_obj_file_name = self._data_root + '/surface_normal_obj.csv'

        fig = plt.figure(figsize=(10,10))

        self._ax = fig.add_subplot(111, projection='3d')
        self._ax.set_title('Object Contacts')

        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')

    def load_data(self):

        return load_data(self._data_file_name)


    def visualize(self, points, color, marker='*', marker_size=3, label=''):

        self._ax.scatter(points[:,0], points[:,1], points[:,2], marker=marker, s=marker_size, color=color, label=label)


    def process_data(self, finger_idx=1):
        '''
        each data point keys: 'obj_state', 'robot_state', 'cp_obj_table', 'cp_obj_obstacle', 'cp_obj_robot'

        robot_state keys: 'pos_js', 'vel_js', 'rea_force_js', 'apl_torque_js', 'pos_ee', 'vel_ee' 
        
        obj_kin_state : each row: pos, ori, vel, omg, lin_acc, ang_acc

        obj_dyn_state : each row: jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque
        
        cp_obj_table keys: 'cp_on_block', 'cp_on_table', 'cn_on_table', 'c_dist', 'cn_force'

        cp_obj_robot each element (= number of fingers) keys: 'cp_on_finger','cp_on_block','cn_on_block','c_dist','cn_force'

        cp_obj_obstacle keys: 'cp_on_block', 'cp_on_obstacle', 'cn_on_obstacle', 'c_dist', 'cn_force'

        '''

        data = self.load_data()

        cp_obj_table_list = []
        cn_obj_table_list = []

        cp_obj_obstacle_list = []
        cn_obj_obstacle_list = []

        cp_obj_robot_list = []
        cn_obj_robot_list = []

        obj_state_list = []


        def convert_point(point, obj_pos, obj_ori):

            # return point

            return self._env.transfer_point_from_world_to_obj(point, obj_pos, obj_ori)[0]


        for data_point in data:

            obj_pos, obj_ori = data_point['obj_kin_state'][:2]
            obj_pos = tuple(obj_pos)
            obj_ori = tuple(obj_ori)

            if data_point['cp_obj_table']['cp_on_block']:

                for po, n in zip(data_point['cp_obj_table']['cp_on_block'], data_point['cp_obj_table']['cn_on_table']):
                    
                    cp_obj_table_list.append(convert_point(np.asarray(po), obj_pos, obj_ori ))
                    cn_obj_table_list.append(convert_point(np.asarray(n), obj_pos, obj_ori ))

            if data_point['cp_obj_obstacle']['cp_on_block']:

                for po, n in zip(data_point['cp_obj_obstacle']['cp_on_block'], data_point['cp_obj_obstacle']['cn_on_obstacle']):
                    
                    cp_obj_obstacle_list.append(convert_point(np.asarray(po), obj_pos, obj_ori ))
                    cn_obj_obstacle_list.append(convert_point(np.asarray(n), obj_pos, obj_ori ))

            if data_point['cp_obj_robot'][finger_idx]['cp_on_block']:

                for po, n in zip(data_point['cp_obj_robot'][finger_idx]['cp_on_block'], data_point['cp_obj_robot'][finger_idx]['cn_on_block']):
                    
                    cp_obj_robot_list.append(convert_point(np.asarray(po), obj_pos, obj_ori ))
                    cn_obj_robot_list.append(convert_point(np.asarray(n), obj_pos, obj_ori ))

            pos, ori, vel, omg, lin_acc, ang_acc = data_point['obj_kin_state']
            jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = data_point['obj_dyn_state']

            obj_state_list.append(np.hstack([pos, pb.getEulerFromQuaternion(tuple(ori)), vel, omg, np.asarray(jnt_reaction_forces), np.asarray(jnt_applied_torque)]))



        
        save_csv_data(self._data_root + 'demo1/cp_obj_table_obj_frame.csv' , cp_obj_table_list)
        save_csv_data(self._data_root + 'demo1/cn_obj_table_obj_frame.csv' , cn_obj_table_list)
        
        save_csv_data(self._data_root + 'demo1/cp_obj_obstacle_obj_frame.csv' , cp_obj_obstacle_list)
        save_csv_data(self._data_root + 'demo1/cn_obj_obstacle_obj_frame.csv' , cn_obj_obstacle_list)
        
        save_csv_data(self._data_root + 'demo1/cp_obj_robot_obj_frame.csv' , cp_obj_robot_list)
        save_csv_data(self._data_root + 'demo1/cn_obj_robot_obj_frame.csv' , cn_obj_robot_list)

        save_csv_data(self._data_root + 'demo1/obj_state_wrld_frame.csv' , obj_state_list)



    def get_force_torques(self):
        sp_obj_frame = load_csv_data(self._sp_obj_file_name)
        sn_obj_frame = load_csv_data(self._sn_obj_file_name)

        force_obj_frame = np.zeros_like(sn_obj_frame)
        torq_obj_frame  = np.zeros_like(sn_obj_frame)

        for k in range(sn_obj_frame.shape[0]):

            force_obj_frame[k,:] = -sn_obj_frame[k,:]
            torq_obj_frame[k,:]  =  np.cross(sp_obj_frame[k,:], -sn_obj_frame[k,:])
        
        return force_obj_frame, torq_obj_frame

    def run(self):

        sp_obj_frame = load_csv_data(self._sp_obj_file_name)
        sn_obj_frame = load_csv_data(self._sn_obj_file_name)

        cp_obj_table_obj_frame = load_csv_data(self._data_root + 'cp_obj_table_obj_frame.csv')
        cn_obj_table_obj_frame = load_csv_data(self._data_root + 'cn_obj_table_obj_frame.csv')

        cp_obj_obstacle_obj_frame = load_csv_data(self._data_root + 'cp_obj_obstacle_obj_frame.csv')
        cn_obj_obstacle_obj_frame = load_csv_data(self._data_root + 'cn_obj_obstacle_obj_frame.csv')

        cp_obj_robot_obj_frame = load_csv_data(self._data_root + 'cp_obj_robot_obj_frame.csv')
        cn_obj_robot_obj_frame = load_csv_data(self._data_root + 'cn_obj_robot_obj_frame.csv')

        obj_state_wrld_frame = load_csv_data(self._data_root + 'obj_state_wrld_frame.csv')

        colors = ['red', 'green', 'blue', 'cyan']

        # self.visualize(points=np.asarray(sp_obj_frame), color='green', label='self')
        # self.visualize(points=np.asarray(cp_obj_table_obj_frame), color='blue', marker_size=28, label='table')
        # self.visualize(points=np.asarray(cp_obj_obstacle_obj_frame), color='cyan', marker_size=28, label='obstacle')
        # self.visualize(points=np.asarray(cp_obj_robot_obj_frame), color='red', marker_size=28, label='robot')

        # self.visualize(points=np.asarray(cp_obj_robot_obj_frame), color='red', marker_size=28, label='robot')
        # self.visualize(points=obj_state_wrld_frame[:,12:15], color='green', marker_size=28, label='force')
        # self.visualize(points=obj_state_wrld_frame[:,15:18], color='blue', marker_size=28, label='torque')

        plt.figure("Force")
        plt.subplot(311)
        plt.plot(0.2*obj_state_wrld_frame[:,12], 'r')
        plt.plot(obj_state_wrld_frame[:,15], 'g')
        plt.plot(obj_state_wrld_frame[:,0], 'b')
        plt.plot(obj_state_wrld_frame[:,3], 'k')

        plt.subplot(312)
        plt.plot(0.2*obj_state_wrld_frame[:,13], 'r')
        plt.plot(obj_state_wrld_frame[:,16], 'g')
        plt.plot(obj_state_wrld_frame[:,1], 'b')
        plt.plot(obj_state_wrld_frame[:,4], 'k')


        plt.subplot(313)
        plt.plot(0.2*obj_state_wrld_frame[:,14], 'r')
        plt.plot(obj_state_wrld_frame[:,17], 'g')
        plt.plot(obj_state_wrld_frame[:,2], 'b')
        plt.plot(obj_state_wrld_frame[:,5], 'k')



        # combined_data = np.vstack([sp_obj_frame,
        #                            cp_obj_table_obj_frame,
        #                            cp_obj_obstacle_obj_frame,
        #                            cp_obj_robot_obj_frame])


        # force_obj_frame, torq_obj_frame = self.get_force_torques()

        # combined_data = np.vstack([sp_obj_frame,
        #                            force_obj_frame,
        #                            torq_obj_frame])


        # # print combined_data.shape

        # # gmm = GMM_GMR(50)
        # # # data: NxM, where N is a number of features and M is a number of recordings or data points
        # # gmm.fit(combined_data.T)

        # # gmm.predict(input)

        # # # computing K-Means with K = 2 
        # centroids,_ = kmeans(combined_data, sp_obj_frame.shape[0])
        # # assign each sample to a cluster
        # idx, _ = vq(combined_data, centroids)

        # self.visualize(points=centroids, color='black', label='kmeans', marker_size=8)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    
    amd = AnalyzeManData()
    amd.process_data()
    amd.run()
        

        

