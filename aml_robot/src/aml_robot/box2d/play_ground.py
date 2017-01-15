import pickle
import matplotlib.pyplot as plt

def load_data(filename):
	
	try:
		pkl_file = open(filename, 'rb')
	except Exception as e:
		raise e

	data = pickle.load(pkl_file)

	pkl_file.close()

	return data




data = load_data('data_test.pkl')

print data[0]['state_end']

img = data[0]['image_rgb_start']
plt.figure(figsize=(8, 8))
plt.imshow(img)

img = data[0]['image_rgb_end']
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.show()