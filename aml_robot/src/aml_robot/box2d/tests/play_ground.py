import pickle
import matplotlib.pyplot as plt

from aml_robot.box2d.core.data_manager import DataManager
from aml_io.convert_tools import string2image


from aml_robot.box2d.push_world.config import config
data_manager = DataManager()
data_manager = data_manager.from_file(filename='data_test_0.pkl',data_folder=config['data_folder_path'])

for i in range(len(data_manager._data)):

    state_start = data_manager.get_sample(i,'state_start')
    state_end = data_manager.get_sample(i,'state_end')

    img_start = string2image(state_start['image_rgb'])
    # img=img.transpose(1,0,2)
    plt.figure(0)
    plt.imshow(img_start)

    plt.figure(1)

    img_end = string2image(state_end['image_rgb'])
    # img=img.transpose(1,0,2)
    plt.imshow(img_end)


    plt.show(block=False)

    raw_input("hello")

# img = data[0]['image_rgb_end']
# plt.figure(figsize=(8, 8))
# plt.imshow(img)