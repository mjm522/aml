
import pickle




def save_data(data,filename):

	output = open(filename, 'wb')

	# Pickle dictionary using protocol 0.
	pickle.dump(data, output)

	# Pickle the list using the highest protocol available.
	# pickle.dump(selfref_list, output, -1)

	output.close()


def load_data(filename):
	pkl_file = open(filename, 'rb')

	data = pickle.load(pkl_file)

	pkl_file.close()

	return data