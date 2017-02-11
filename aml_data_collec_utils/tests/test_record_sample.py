import aml_data_collec_utils

from aml_data_collec_utils.record_sample import DataManager

data_man = DataManager(data_name_prefix='push_data')

data = data_man.read_data(1)

print type(data)

for sample in data:

	print "##########################################################################################################"

	keys = sample.keys()

	for key in sample:

		print "****************************************************************************************************"
		print "Key \t", key
		print "Value \t", sample[key]
		print "****************************************************************************************************"




