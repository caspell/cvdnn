import numpy as np
import cv2
import time
import os

# cap = cv2.VideoCapture('/data/share/nfs/40/video.h264')
#cap = cv2.VideoCapture('/home/mhkim/data/video/MVI_0043.AVI')
#cap = cv2.VideoCapture('rtsp://192.168.1.40:8554/')
# cap = cv2.VideoCapture('http://192.168.1.40:8080/?action=stream')
# cap = cv2.VideoCapture('/home/mhkim/data/video/sampho2.mp4')

filename = 'video/sampho2.mp4'

cap = cv2.VideoCapture(os.path.join('/home/mhkim/data',filename))



# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def draw_flow1(old_gray, frame_gray, p0 , mask):

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame , mask)

    cv2.imshow('frame' , img)


    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    return ( old_gray , p0 )

def draw_flow2(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def imageResize (frame , rate=1.) :
    h , w = frame.shape[:2]
    dim = (int(w * rate), int(h * rate))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

imgCrop = (lambda f : f[35:] )

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def imageSharpening (frame) :
    frame = imageResize(frame, 0.6)
    return cv2.filter2D(frame, -1, kernel)

old_frame = imgCrop(old_frame)



old_frame = imageSharpening(old_frame)


old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    ret,frame = cap.read()

    frame = imgCrop(frame)

    frame = imageSharpening(frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # optical flow - test 1
    # old_gray , p0 = draw_flow1(old_gray, frame_gray, p0, mask)


    # optical flow - test 2
    # flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 13, 15, 3, 5, 1.2, 0)

    # cv2.imshow('flow', draw_flow2(frame_gray, flow))
    cv2.imshow('flow', frame_gray)


    # time.sleep(0.2)
cv2.destroyAllWindows()
cap.release()