import numpy as np
import cv2
import time , math
import matplotlib.pyplot as plt
from cassandra.cluster import Cluster

_bytes = np.random.random([480, 1040, 2])

_bytes = _bytes.astype(np.float64)

def insert () :

    cluster = Cluster(['192.168.1.50'])

    # session = cluster.connect('sampyo_data')
    session = cluster.connect('test_data')


    bytes_data = bytearray(_bytes)

    to_strings = (lambda x: '%s' % format(x))

    shape_str = ','.join(map(to_strings, _bytes.shape))

    try :

        insert_data = dict(frameCount=10, data=bytes_data, dtype=str(_bytes.dtype), shape=shape_str)

        # print ( insert_data)
        session.execute("""
            insert into frames ( partition_key , frame_num , data , dtype , shape )
            values ( 1, %(frameCount)s , %(data)s , %(dtype)s , %(shape)s ) """
                        , insert_data)

        # '''
        # session.execute("""
        #     insert into frames ( partition_key , frame_num , data , dtype , shape ) values (1, %s, %s, %s, %s)
        # """, (1, 4, bytes_data, str(dtype), shape_str))
        # session.execute("""
        #     insert into frames ( partition_key , frame_num , data , dtype , shape ) values (1, %s, %s, %s, %s)
        # """, (1, bytes_data, str(dtype), shape_str))
        # '''

    finally :
        cluster.shutdown()


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

def draw_flow_lines (flow, step=16):

    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = np.zeros((h, w, 3), np.uint8)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

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


def read ():

    cluster = Cluster(['192.168.1.50'])

    session = cluster.connect('sampyo_data')


    try:

        query = "select * from frames where partition_key = 1 order by frame_num ASC limit 100 "

        ff = session.execute_async(query)

        rows = ff.result()

        values = []

        for r in rows:
            print (r.frame_num)

            data = r.data

            frame = np.frombuffer(data, dtype=np.float32)#.astype(np.float64)

            # frame = np.fromstring(data, dtype='float32')
            frame = frame.reshape(480,1040,2)

            lines = get_draw_flow_lines(frame)

            value = get_avg_travel_distance(lines)

            values.append(value)

            # max1 = np.max(frame[:, :, 0])
            # max2 = np.max(frame[:, :, 1])

            # print ( value )

            # flows = draw_flow_lines(frame)

            # print ( flows )

            # cv2.imshow('frame-flow', flows)

            # time.sleep(1.)

            # print ( flows )

            # break

        fig, ax = plt.subplots()

        print ( values)

        ax.plot(values)

        plt.show()

    finally:
        cv2.destroyAllWindows()
        cluster.shutdown()



def test () :
    bytes = np.random.random([480, 1040, 2])

    print (len(bytearray(bytes.astype(np.uint8))))

    print (len(bytearray(bytes.astype(np.float32))))

    print (len(bytearray(bytes.astype(np.float64))))


if __name__ == '__main__' :
    # insert()
    read()
    #test()