import aml_data_collec_utils

import cv2

from aml_data_collec_utils.core.data_manager import DataManager

data_man = DataManager(data_name_prefix='test_push_data')


from aml_io.convert_tools import image2string, string2image

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

quit = False

key = 0

data = data_man.read_data(1)


print data

sample_idx = 0

print "NUMBER OF SAMPLES: ", len(data)

while not quit:

    sample = data[sample_idx]



    image = string2image(sample.get(0,['rgb_image'])[0])

    
    cv2.imshow("RGB Image Before", image)


    image = string2image(sample.get(-1,['rgb_image'])[0])

    print "TERMINAL", sample.get(-1,['terminal'])
    print "SAMPLE SIZE ", sample.size()

    cv2.imshow("RGB Image After", image)

    print "SAMPLE_ID: \t", sample._id
    print "STATUS: \t", sample._is_valid
    print "START STATE ", sample.get_keys()
    print "FINAL STATE ", sample.get_keys()
    print "Start location of the box \n", sample.get(0,['task_state'])
    print "Start location of the box \n", sample.get(0,['task_state'])
    print "Push action \n", sample.get(0,['task_action'])
    print "End location of the box \n", sample.get(-1,['task_state'])

    for i in range(sample.size()):
        cv2.imshow("Image sequence",  string2image(sample.get(i,['rgb_image'])[0]))
        print "Image sequence index ", i, " sample id ", sample.get_id()
        cv2.waitKey(0)

    

    key = cv2.waitKey(0)

    print key
    if key == 65363 or key == 1113939:
    	sample_idx += 1
    elif key == 65361 or key == 1113937:
    	sample_idx = max(0,sample_idx-1)

    if key == 27 or sample_idx >= len(data):
        quit = True



print "BYE!"


