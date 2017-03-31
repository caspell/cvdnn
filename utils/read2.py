import numpy as np
import matplotlib.pyplot as plt

# video_path, data_path = '/home/mhkim/data/sampyo/long-1.mp4', '/home/mhkim/data/numpy/sampyo/data_long'
# video_path, data_path = '/home/mhkim/data/sampyo/short-1.mp4', '/home/mhkim/data/numpy/sampyo/data_short_1'
# video_path, data_path = '/home/mhkim/data/sampyo/short-2.mp4', '/home/mhkim/data/numpy/sampyo/data_short_2'


with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data', 'r') as fd :
    arr = np.load(fd)

print ( np.shape(arr) )

# for ar in arr :
#     print ( ar)

print (len(arr) / 3600)

fig, ax = plt.subplots()

ax.plot(arr[:100])

plt.show()
