import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

STEP = 30
RANGE_NUM = 9000
def get_linvalue(data, size, focus=0):

    values = []
    xaxis = [x for x in range(STEP)]

    for index in range(size):
        begin = STEP * index
        end = begin + STEP

        value = data[begin:end]

        value = value[:, focus]

        A = np.vstack([xaxis, np.ones(STEP)]).T

        m, c = lin.lstsq(A, value)[0]

        _x = m * np.array(xaxis) + c

        # values.append(_x)
        values.append(c)

    return values

def draw_line (data , ax) :

    size = 300

    _xaxis = [x for x in range(size)]
    _yaxis = np.full_like(_xaxis, 0)

    # print (_xaxis, _yaxis)

    values1 = get_linvalue(data, size, 0)
    # values2 = get_linvalue(data, size, 1)
    values3 = get_linvalue(data, size, 3)

    # ax.errorbar(_xaxis, _yaxis, lolims=1, yerr=25., linestyle='dotted')

    lines1 = ax.plot(_xaxis, values1, 'r', color='red', label='average')

        # , xerr=xerr, yerr=yerr, lolims=lolims, linestyle=ls)

    # lines2 = ax.plot(_xaxis, values2, 'r', color='blue', label='max')

    # ax.errorbar(_xaxis, _yaxis, lolims=1, yerr=25., linestyle='dotted')

    lines3 = ax.plot(_xaxis, values3, 'r', color='green', label='moved average')

    ax.grid(True)

def main () :

    arr = load_avg_data()

    plot_count = 10
    # plot_count = 30

    fig = plt.figure(1)

    # arr = arr[90000:]

    major_ticks = np.arange(0, 300, 5)
    minor_ticks = np.arange(0, 300, 1)

    for i in range(plot_count) :
        begin = RANGE_NUM * i
        end = begin + RANGE_NUM

        data = arr[begin:end]

        ax = fig.add_subplot(plot_count, 1, i + 1)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        draw_line(data , ax)

    plt.legend()
    plt.show()

    # plt.plot(xaxis, data)
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__' :
    main()