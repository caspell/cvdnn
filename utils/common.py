import cv2
import numpy as np
_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def image_precondition (frame , sharpening=-1, rate=1. , crop_height=None, crop_width=None) :
    if crop_height != None or crop_width != None :
        frame = imageCrop(frame, crop_height, crop_width)

    # frame = imageResize(frame, rate )

    frame = imageSharpening ( frame , sharpening )

    return frame

def imageSharpening (frame , sharpening=-1) :
    return cv2.filter2D(frame, sharpening, _kernel)

def imageResize (frame , rate=1.) :
    h , w = frame.shape[:2]
    dim = (int(w * rate), int(h * rate))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def imageCrop (frame, crop_height=None,crop_width=None) :

    if crop_height != None and crop_width != None :
        return frame[0:crop_height, 0:crop_width, :]
    elif crop_height != None :
        return frame[0:crop_height, :]
    else :
        return frame[:, 0:crop_width, :]

