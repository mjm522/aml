import StringIO
from PIL import Image
import cv2


def image2string(image_in, fmt = 'png'):

    out = image_in
    if type(image_in) != type(""):
        pil_img = Image.fromarray(image_in)
        output = StringIO.StringIO()
        pil_img.save(output,format=fmt)
        out = output.getvalue()

    return out

def string2image(str_image_in):
    '''
    expects a string array
    '''
    nparr = np.fromstring(str_image_in, np.uint8)
    out = cv2.imdecode(nparr, 1)#cv2.CV_LOAD_IMAGE_COLOR
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out