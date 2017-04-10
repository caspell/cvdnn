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

from cassandra.cluster import Cluster

import numpy as np
import cv2
import cassandra
import os
from utils.common import image_precondition
import math

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import time


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = 255#ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow_lines2 (flow, step=16):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = 255#ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow_lines (flow, step=16):

    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = np.zeros((h, w, 3), np.uint8)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

def get_draw_flow_lines(flow, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = np.zeros((h, w, 3), np.uint8)
    # cv2.polylines(vis, lines, 0, (0, 255, 0))
    return lines

def get_avg_travel_distance ( lines , step=1):

    t1 , t2 = lines[:, :1] , lines[:, 1:]

    num = 0.
    max = 0.
    min = 0.
    diff = 0.
    diff_count = 0
    avg_diff = 0

    get_distance = (lambda x1, y1, x2, y2 : math.sqrt((x2-x1)**2 + (y2-y1)**2))

    for (x1, y1), (x2, y2) in lines:
        distance = get_distance(x1, y1, x2, y2)
        num = num + distance
        if max < distance :
            max = distance
        if min > distance:
            min = distance
        if x1 <> x2 and y1 <> y2 :
            diff = diff + distance
            diff_count = diff_count + 1

    avg = num / len(lines)

    if diff_count > 0 :
        avg_diff = diff / diff_count

    return [avg , max, min , avg_diff]


def get_frame (frame, crop_height=None) :

    _frame = image_precondition(frame, crop_height=crop_height)

    return cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

def analysis (video_path , data_path) :
    import sys
    #print(__doc__)


    # cluster = Cluster(['192.168.1.50'])
    #
    # session = cluster.connect('sampyo_data')

    FLOW_STEP = 16

    # try:
    #     fn = sys.argv[1]
    # except IndexError:
    #     fn = 0

    # cam = video.create_capture(fn)
    cam = cv2.VideoCapture(video_path)

    ret, prev = cam.read()

    h , w = np.shape(prev)[0:2]

    height = h - 60

    prevgray = get_frame(prev, crop_height=height)

    show_hsv = False
    show_lines = False
    save_data = True
    cur_glitch = prev.copy()

    frameCount = 0

    values = []

    to_strings = (lambda x: '%s' % format(x))


    # cursor = session.execute_async('select max(frame_num) as frame_num from frames')

    # rows = cursor.result()
    #
    # last_frame_num = 0
    #
    # for r in rows:
    #     last_frame_num = r.frame_num
    #     # print ( last_frame_num)
    #     break

        # cluster.shutdown()
        # return

    try :
        while True:

            print ("frameCount : " , frameCount)

            ret, frame = cam.read()

            if not ret :
                break

            # if frameCount < last_frame_num:
            #     if frameCount == last_frame_num - 1:
            #         prevgray = get_frame(frame, crop_height=height)
            #
            #     frameCount = frameCount + 1
            #     continue

            gray = get_frame(frame, crop_height=height)

            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            prevgray = gray


            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(gray, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('flow', draw_flow(gray, flow, FLOW_STEP))

            if show_hsv:
                cv2.imshow('flow HSV', draw_hsv(flow))

            if show_lines :
                # cv2.imshow('flow flow lines', draw_flow_lines(flow, FLOW_STEP))
                lines = get_draw_flow_lines(flow, FLOW_STEP)
                value = get_avg_travel_distance(lines)

            if save_data :
                # lines = get_draw_flow_lines(flow, FLOW_STEP)
                # value = get_avg_travel_distance(lines)
                # values.append(flow)

                _bytes = flow.astype(np.float32)

                data = bytearray(_bytes)

                shape_str = ','.join(map(to_strings, _bytes.shape))

                insert_data = dict(frameCount=frameCount , data=data , dtype=str(_bytes.dtype), shape=shape_str)

                #print ( insert_data)

                # session.execute(""" insert into frames ( partition_key , frame_num , data , dtype , shape )
                #                     values ( 1, %(frameCount)s , %(data)s , %(dtype)s , %(shape)s ) """
                #                     , insert_data, 30000)

            # cv2.putText()

            ch = cv2.waitKey(5)
            #
            # if ch == 82 :
            #     #up
            #     FLOW_STEP = FLOW_STEP + 1
            # if ch == 84 :
            #     #down
            #     FLOW_STEP = FLOW_STEP - 1
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_lines = not show_lines
                print('flow line visualization is', ['off', 'on'][show_lines])

            frameCount = frameCount + 1

    finally :
        print ('finished~')
        cv2.destroyAllWindows()
        # cluster.shutdown()
        cam.release()

    # with open(data_path, 'w') as fd :
    #     np.save(fd, values)


if __name__ == '__main__':

    video_path, data_path = '/data/samba/anonymous/sampyo_6_hour.mp4', '/home/mhkim/data/numpy/sampyo/sampyo_frame_data'
    analysis(video_path , data_path)

    # video_path, data_path = '/home/mhkim/data/sampyo/short-1.mp4', '/home/mhkim/data/numpy/sampyo/data_short_1'
    # analysis(video_path, data_path)
    #
    # video_path, data_path = '/home/mhkim/data/sampyo/short-2.mp4', '/home/mhkim/data/numpy/sampyo/data_short_2'
    # analysis(video_path, data_path)