import numpy as np
import cv2
import time , math

from utils.common import image_precondition

cap = cv2.VideoCapture('/data/samba/anonymous/sampyo_6_hour.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 3000,
                       qualityLevel = 0.1,
                       minDistance = 1,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (30, 30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_frame (frame, crop_height=None) :

    _frame = image_precondition(frame, crop_height=crop_height)

    return cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)


# Create some random colors
#color = np.random.randint(0,255,(3000,3))
color = np.full((3000,3), 255)

ret, old_frame = cap.read()

h, w = np.shape(old_frame)[0:2]

height = h - 60

frameCount = 0

old_gray = get_frame(old_frame, crop_height=height)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_gray)

RESET_FRAME = 30

while(1):
    # math.fmod()
    # print ( divmod(RESET_FRAME , frameCount + 1)[1])

    if divmod(frameCount + 1, RESET_FRAME)[1] == 0:
        print ('reset')
        mask = np.zeros_like(old_gray)

    ret,frame = cap.read()

    frame_gray = get_frame (frame, crop_height=height)


    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #
    # frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_GRADIENT, kernel)


    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None :
        break

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)


    # print ( np.shape(frame) )
    # print (np.shape(mask))

    img = cv2.add(frame_gray , mask)


    ######################
    cv2.imshow('frame', img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

    p0 = good_new.reshape(-1,1,2)

    frameCount = frameCount + 1

    # time.sleep(0.2)

cv2.destroyAllWindows()
cap.release()