
import StringIO
import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

def image2string(image_in, fmt = 'png'):

	out = image_in
	if type(image_in) != type(""):
	    pil_img = Image.fromarray(image_in)
	    output = StringIO.StringIO()
	    pil_img.save(output,format=fmt)
	    out = output.getvalue()

	return out

def dimage2string(image_in, fmt='tif'):
	# print "***********************************HERE", type(image_in[0,0])
	pil_img = Image.fromarray(image_in)
	output = StringIO.StringIO()
	# im = Image.fromstring('I;16',image_in.shape,image_in.tostring())
	# im.save('test_16bit.tif')
	pil_img.save('/home/baxter_gps/catkin_workspaces/baxter_ws/src/aml/newimage.spi', format='SPIDER')

def string2image(str_image_in):
    '''
    expects a string array
    '''
    nparr = np.fromstring(str_image_in, np.uint8)
    out = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

def rosimage2openCVimage(ros_image):
	bridge = CvBridge()
	try:
		cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
	except CvBridgeError as e:
		print(e)
	rgb_image = np.array(cv_image, dtype=np.uint8)
	return rgb_image
