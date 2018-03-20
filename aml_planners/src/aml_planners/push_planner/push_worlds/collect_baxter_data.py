import os
import tf, rospy
import scipy.misc
import numpy as np
import pandas as pd
from aml_perception.camera_sensor import CameraSensor
from controller_ros.baxter_push_world import BaxterPushWorld



class CollectBaxterData():

    def __init__(self, config):
        self._config = config
        self._camera_sensor = CameraSensor()
        self._world =  BaxterPushWorld()
        self._df = pd.DataFrame(columns=['xi','yi','thetai',
                                         'action',
                                         'xf','yf','thetaf'])
        self._push_idx = 0


    def get_push_action(self, specific_side=None):
        push_action_left  = np.random.uniform(0, 0.25) #me left when i face the robot
        push_action_right = np.random.uniform(0.25, 0.5)
        push_action_front = np.random.uniform(0.5, 0.75)
        push_action_back  = np.random.uniform(0.75, 1.)
         
        push_actions = [push_action_left, push_action_right, push_action_front, push_action_back]

        if specific_side is None:
            return push_actions[np.random.randint(0,4)]
        else:
            return push_actions[specific_side]


    def save_image(self, image_index):
        curr_image = copy.deepcopy(self._camera_sensor._curr_rgb_image)
        if curr_image is not None:
            filename = self._config['exp_image_folder'] + str(image_index) + '.jpg'
            scipy.misc.imsave(filename, curr_image)

    def save_csv(self):

        self._df.to_csv(self._config['filename'])


    def collect_push(self):

        state0 = self._world.state()

        action = [self.get_push_action(self._config['side_to_push'])]

        for i in range(self._config['sequence_push']): 
            ## Send action to world
            sucess = self._world.apply_push(action)

        statef = self._world.state()
        
        tmp = pd.DataFrame(np.reshape(np.r_[state0[:3],action[0],statef[:3]],(1,7)), columns=['xi','yi','thetai',
                                                                                              'action',
                                                                                              'xf','yf','thetaf'])
        print "State initial \t", np.round(state0,3)
        print "Action \t", action[0]
        print "State final \t", np.round(statef, 3)
        choice = raw_input("Want to save that push? (y/n)")

        if choice == 'y':
            self._df = self._df.append(tmp, ignore_index=True)
            self._push_idx += 1


    def collect_learning_data(self):
        while self._push_idx < self._config['total_pushes']:

            for k in range(self._config['push_per_location']):
                self.collect_push()

            choice = raw_input("Continue in same location? (y/n)")

            if choice == 'y':
                continue

        self.save_csv()




def main(side_to_push=None):
    rospy.init_node("baxter_data_collection_node")

    sides = ['0025', '2550', '5075', '75100']

    print "*************************************************"
    raw_input("Remembered to rename the file?")
    print "*************************************************"

    config = {
    'sequence_push':1,
    'push_per_location':2,
    'total_pushes':10,
    'side_to_push':side_to_push, #make it none to push random sides
    'filename':os.environ['MPPI_DATA_DIR'] + 'baxter_push_data_side_' + sides[side_to_push] + '_12.csv',
    }

    cbd = CollectBaxterData(config)

    cbd.collect_learning_data()


if __name__ == '__main__':
    main(side_to_push=3)