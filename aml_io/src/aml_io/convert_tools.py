
import StringIO
import Image
import numpy as np
import cv2


def image2string(image_in, fmt = 'png'):

    pil_img = Image.fromarray(image_in)
    output = StringIO.StringIO()
    pil_img.save(output,fmt)
    out = output.getvalue()

    return out

def string2image(str_image_in):
    nparr = np.fromstring(str_image_in, np.uint8)
    out = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

    return out