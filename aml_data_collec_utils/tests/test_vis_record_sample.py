import aml_data_collec_utils

import cv2

import numpy as np

from aml_data_collec_utils.core.data_manager import DataManager

data_man = DataManager(data_name_prefix='test_push_data')

from aml_io.convert_tools import image2string, string2image

import sys

# data = data_man.read_data(1)

# print type(data)

# for sample in data:

#     print "##########################################################################################################"

#     keys = sample.keys()

#     for key in sample:

#         print "****************************************************************************************************"
#         print "Key \t", key
#         print "Value \t", sample[key]
#         print "****************************************************************************************************"


if len(sys.argv) < 2:
    print "missing data index, e.g. python test_vis_sample 1"
    sys.exit(1)

data_idx = int(sys.argv[1])

quit = False

key = 0

data = data_man.read_data(data_idx)

print data

sample_idx = 0
sample_data_point_idx = 0

print "NUMBER OF SAMPLES: ", len(data)

while not quit:

    sample = data[sample_idx]

    image = string2image(sample.get(0,['rgb_image'])[0])
 
    cv2.imshow("RGB Image Before", image)

    image = string2image(sample.get(-1,['rgb_image'])[0])

    print "TERMINAL", sample.get(-1,['terminal'])
    print "SAMPLE SIZE ", sample.size

    cv2.imshow("RGB Image After", image)

    print "SAMPLE_ID: \t", sample._id
    print "STATUS: \t", sample._is_valid
    print "SAMPLE KEYS ", sample.get_keys()
    # print "START STATE ", sample._contents[0]
    # print "FINAL STATE ", sample._contents[-1]
    print "Start location of the box \n", sample.get(0,['box_pos'])
    print "Push action \n", sample.get(0,['task_action'])
    print "End location of the box \n", sample.get(-1,['box_pos'])
    print "Box tracking good Before", sample.get(0,['box_tracking_good'])
    print "Box tracking good After", sample.get(-1,['box_tracking_good'])

    sample_is_valid = np.linalg.norm(sample.get(-1,['box_pos'])[0] - sample.get(0,['box_pos'])[0]) > 1e-7 and sample.size > 2

    print "Sample is valid ", sample_is_valid

    #for i in range(sample.size()):
    cv2.imshow("Image sequence",  string2image(sample.get(sample_data_point_idx,['rgb_image'])[0]))
    print "Image sequence index ", sample_data_point_idx, " sample id ", sample.get_id()
    cv2.waitKey(0) 

    key = cv2.waitKey(0)

    print key
    if key == 65363 or key == 1113939:
        sample_idx = (sample_idx + 1)%len(data)
        sample_data_point_idx = 0
    elif key == 65361 or key == 1113937:
        sample_idx = max(0,sample_idx-1)
        sample_data_point_idx = 0
    elif key == 1048678:
        sample_data_point_idx = (sample_data_point_idx + 1)%sample.size

    if key == 27 or key == 1048603 or sample_idx >= len(data):
        quit = True

print "BYE!"


