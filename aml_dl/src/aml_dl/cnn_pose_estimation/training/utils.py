import aml_robot
from aml_io.io_tools import save_data, load_data
import cv2
import rospy

def show_image(image):

    cv2.imshow("RGB Image window", image)

    cv2.waitKey(0)


def show_image(idx, data):

    image = data[idx]['rgb_image']

    show_image(image)


def load_data_tf(filename):
    data = load_data(filename)

    return data


def prepare_data_tf(data,img_width, img_height):

	# Resize to appropriate dimensions, transpose, and flatten
	prepared_img = lambda img: np.transpose(cv2.resize(img, (img_width, img_height)),(2,1,0)).flatten()

	prepared_data = [ np.r_[s['position'],s['velocity'], prepared_img(s['rgb_image'])] for s in data ]


	return prepared_data


def draw_features(features,image_data):

    try:
        (rows,cols,channels) = image_data.shape
        i_feat = 0
        while i_feat < len(features):
            cx = cols/2 + int(float(features[i_feat]*cols)/2)
            cy = rows/2 + int(float(features[i_feat+1]*rows)/2)
            cv2.circle(image_data, (cx,cy), 3, 255)
            i_feat += 2
    except Exception as e:
            print(e)

    return image_data


