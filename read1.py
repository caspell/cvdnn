import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# import confluent_kafka as kafka
#
# conf = {'bootstrap.servers': 'localhost:9092',  'default.topic.config': {'acks': 'all'}}
#
# p = kafka.Producer(conf)
#
#
# # video_path, data_path = '/home/mhkim/data/sampyo/long-1.mp4', '/home/mhkim/data/numpy/sampyo/data_long'
# # video_path, data_path = '/home/mhkim/data/sampyo/short-1.mp4', '/home/mhkim/data/numpy/sampyo/data_short_1'
# # video_path, data_path = '/home/mhkim/data/sampyo/short-2.mp4', '/home/mhkim/data/numpy/sampyo/data_short_2'
#
#
# with open('/home/mhkim/data/numpy/sampyo/data_long', 'r') as fd :
#     arr = np.load(fd)
#
# print ( np.shape(arr) )
#
# for ar in arr :
#     # print ( ar)
#
#     p.produce('testing', ar.tobytes())
#
#     p.flush()
#
#
# fig, ax = plt.subplots()
#
# ax.plot(arr[:100,0::3])
#
# plt.show()
# #
