import pickle
import matplotlib.pyplot as plt

from data_manager import DataManager



data_manager = DataManager()
data_manager =data_manager.from_file('data_test.pkl')


print data_manager.get_sample(0,'state_end')

# img = data[0]['image_rgb_start']
# plt.figure(figsize=(8, 8))
# plt.imshow(img)

# img = data[0]['image_rgb_end']
# plt.figure(figsize=(8, 8))
# plt.imshow(img)
# plt.show()