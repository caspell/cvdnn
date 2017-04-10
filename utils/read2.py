import numpy as np
import matplotlib.pyplot as plt

# video_path, data_path = '/home/mhkim/data/sampyo/long-1.mp4', '/home/mhkim/data/numpy/sampyo/data_long'
# video_path, data_path = '/home/mhkim/data/sampyo/short-1.mp4', '/home/mhkim/data/numpy/sampyo/data_short_1'
# video_path, data_path = '/home/mhkim/data/sampyo/short-2.mp4', '/home/mhkim/data/numpy/sampyo/data_short_2'

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

def main () :

    arr = load_avg_data()

    print ( np.shape(arr) )

    print ( arr.dtype )

    range_num = 9000

    fig = plt.figure(1)

    step = 10

    _range = 30

    secondline = [x * _range for x in range(range_num // _range)]

    yaxis = [ 25 for x in range(range_num // _range)]


    for c in range(step) :
        _begin = c * range_num
        _end = _begin + range_num
        data = arr[_begin : _end,:]

        ax = fig.add_subplot(step , 1 , c + 1)
        # ax.errorbar(range_num, range_num, yerr=30, fmt='-o')

        ax.errorbar(secondline, yaxis, yerr=25, uplims=True, lolims=False)
        ax.plot(data)

    plt.show()

    #
    # data1 = arr[:300]
    # data2 = arr[300:600]
    #
    # # for ar in arr :
    # #     print ( ar)
    #
    # fig = plt.figure(1)
    #
    # ax1 = fig.add_subplot(211)
    #
    # ax1.plot(data1)
    #
    # data2 = data2.astype('uint8')
    #
    # ax2 = fig.add_subplot(212)
    #
    # ax2.plot(data2)
    #
    # plt.show()


if __name__ == '__main__' :
    main()