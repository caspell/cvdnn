import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

def load_avg_train_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_avg_train_result', 'r') as fd :
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

def draw_line (ax, data , xaxis) :

    size = len(xaxis)

    values1 = get_linvalue(data, size, 0)

    values3 = get_linvalue(data, size, 3)

    ax.plot(xaxis, values1, 'r', color='red', label='average')

    ax.plot(xaxis, values3, 'r', color='blue', label='moved average')

def main () :

    arr = load_avg_data()

    result = load_avg_train_data()

    plot_count = 10

    fig = plt.figure(1)

    arr = arr[90000:90000+90000]

    line_limit = 300

    major_ticks = np.arange(0, line_limit, 5)
    minor_ticks = np.arange(0, line_limit, 1)

    _xaxis = [x for x in range(line_limit)]

    for i in range(plot_count) :

        begin = RANGE_NUM * i
        end = begin + RANGE_NUM

        data = arr[begin:end]

        oneline = i * line_limit

        pred_value = result[oneline:oneline + line_limit] * 6 - 5

        ax = fig.add_subplot(plot_count, 1, i + 1)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        draw_line(ax , data , _xaxis)

        ax.plot(_xaxis, pred_value, 'r', color='green', label='predict')

        ax.grid(True)

    plt.legend()
    plt.show()

if __name__ == '__main__' :
    main()

    # result = load_avg_train_data()
    #
    # print ( result)