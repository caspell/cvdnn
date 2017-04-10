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




_data = [
            [[60,  80] , [120, 137]],
            [[145, 165] , [190, 212]],
            [[218, 240] , [298, 317]],
            [[328, 342] , [410, 442]],
            [[453, 467] , [610, 582]],
            [[588, 608] , [635, 658]],
            [[663, 683] , [725, 746]],
            [[757, 771] , [801, 879]],
            [[885, 904] , [940, 960]],
            [[967, 985] , [1030, 1138]],
            [[1147, 1162] , [1195, 1228]],
            [[1230, 1250] , [1280, 1318]],
            [[1330, 1343] , [1385, 1405]],
            [[1533, 1547] , [1572, 1588]],
            [[1881, 1902] , [1951, 1976]],
            [[1987, 2001] , [2028, 2047]],
            [[2053, 2071] , [2100, 2152]],
            [[2271, 2291] , [2314, 2334]],
            [[2343, 2364] , [2386, 2410]],
            [[2420, 2435] , [2467, 2502]],
            [[2780, 2794] , [2827, 2844]]
        ]

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr