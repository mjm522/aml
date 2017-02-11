import aml_data_collec_utils

import cv2

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







quit = False

key = 0

data = data_man.read_data(1)
sample_idx = 0

while not quit:

    sample = data[sample_idx]


    image = sample['state_before']['rgb_image']
    cv2.imshow("RGB Image Before", image)

    image = sample['state_after']['rgb_image']
    cv2.imshow("RGB Image After", image)

    print "SAMPLE_ID: ", sample['sample_id'], "SAMPLE_IDX: ", sample_idx

    key = cv2.waitKey(0)

    sample_idx += 1

    if key == 27 or sample_idx >= len(data):
        quit = True



print "BYE!"


