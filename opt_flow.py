#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import cassandra

def draw_flow(img, flow, step=16):
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

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow_lines (img, flow, step=16):
    h, w = img.shape[:2]

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)

    lines = np.int32(lines + 0.5)

    vis = np.zeros((h, w, 3), np.uint8)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

IMAGE_SHARPEN_NUM = -1

def imageSharpening (frame) :
    return cv2.filter2D(frame, IMAGE_SHARPEN_NUM, kernel)

if __name__ == '__main__':
    import sys
    print(__doc__)

    fn = '/home/mhkim/data/video/sampho2.mp4'

    # try:
    #     fn = sys.argv[1]
    # except IndexError:
    #     fn = 0

    # cam = video.create_capture(fn)
    cam = cv2.VideoCapture(fn)

    ret, prev = cam.read()

    prev = imageSharpening(prev)

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_lines = False
    cur_glitch = prev.copy()

    frameCount = 0

    while True:

        ret, img = cam.read()

        img = imageSharpening(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prevgray = gray

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(gray, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('flow', draw_flow(gray, flow))


        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_lines :
            cv2.imshow('flow flow lines', draw_flow_lines(gray, flow))
        # cv2.putText()

        ch = cv2.waitKey(5)

        # print ( 'IMAGE_SHARPEN_NUM : ' , IMAGE_SHARPEN_NUM )

        if ch == 82 :
            #up
            IMAGE_SHARPEN_NUM = IMAGE_SHARPEN_NUM + 1
            if IMAGE_SHARPEN_NUM > -1 :
                IMAGE_SHARPEN_NUM = -1
        if ch == 84 :
            #down
            IMAGE_SHARPEN_NUM = IMAGE_SHARPEN_NUM - 1
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_lines = not show_lines
            print('flow line visualization is', ['off', 'on'][show_lines])

    cv2.destroyAllWindows()
