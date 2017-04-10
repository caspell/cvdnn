import numpy as np
import cv2
import time

from utils.common import image_precondition

# cap = cv2.VideoCapture('/data/share/nfs/40/video.h264')
#cap = cv2.VideoCapture('/home/mhkim/data/video/MVI_0043.AVI')
#cap = cv2.VideoCapture('rtsp://192.168.1.40:8554/')

cap = cv2.VideoCapture('/data/samba/anonymous/sampyo_6_hour.mp4')
# cap = cv2.VideoCapture('http://192.168.1.40:8080/?action=stream')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_frame (frame, crop_height=None) :

    _frame = image_precondition(frame, crop_height=crop_height)

    return cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)


# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

# imgCrop = (lambda f : f[35:] )
#
# old_frame = imgCrop(old_frame)

h, w = np.shape(old_frame)[0:2]

height = h - 60

frameCount = 0
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

old_gray = get_frame(old_frame, crop_height=height)

# old_frame = get_frame(old_frame, crop_height=height)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_gray)

while(1):

    ret,frame = cap.read()

    # frame = imgCrop(frame)

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_gray = get_frame (frame, crop_height=height)

    # frame = get_frame(frame, crop_height=height)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None :
        break

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    # print ( np.shape(frame) )
    # print (np.shape(mask))

    # break

    img = cv2.add(frame_gray , mask)

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