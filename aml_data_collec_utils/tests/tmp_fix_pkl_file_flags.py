import os
from aml_data_collec_utils.core.sample import Sample
from aml_data_collec_utils.core.data_manager import DataManager

import numpy as np


data_folder_path = os.environ['AML_DATA'] + '/aml_dl/push_data_post_processed/'
data_name_prefix = 'test_push_data'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


def check_sample(sample):
    sample_is_valid = np.linalg.norm(sample.get(-1,['box_pos'])[0] - sample.get(0,['box_pos'])[0]) > 1e-7 and sample.size > 2
    return sample_is_valid

def main():
    data_file_range = range(1,632)

   

    for data_idx in data_file_range:

        data_man = DataManager(data_name_prefix='test_push_data')
        data_list = data_man.read_data(data_idx)

        if data_list is None:
            print "The data is None, proceeding to next one"
            continue
        
        print data_idx

        ids_to_keep = []
        num_samples = len(data_list)

        print num_samples
        for k in range(num_samples):
            sample = data_list[k]
            print "Sample number \t", k
            if check_sample(sample):
                print "Check sample returned is valid"
                ids_to_keep.append(k)
            else:

                print "Status according to sample \t", sample.is_valid()
                sample.set_valid(False)
                

        data_man._data = [data_list[s_idx] for s_idx in ids_to_keep]

        if len(data_man._data) > 0:
            data_man._data_folder_path = data_folder_path
            data_man._data_idx = data_idx

            data_man.write_data()



if __name__ == '__main__':
    main()









