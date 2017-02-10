#!/usr/bin/env python

import cv2

import aml_robot

from aml_io.io_tools import save_data, load_data

data = load_data('../data/push_data_sample_00.pkl')
print data
# image = data[0]['rgb_image']

# print(image)


# cv2.imshow("RGB Image window", image)

# cv2.waitKey(0)