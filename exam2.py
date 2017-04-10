import numpy as np
import numpy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt
from pandas.compat import range, lrange, lmap, map, zip
from pandas.tools.plotting import scatter_matrix,autocorrelation_plot

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

def lin1 () :

    arr = load_avg_data()

    range_num = 30

    xaxis = [x for x in range(range_num)]

    data = arr[0: range_num, 0]

    A = np.vstack([xaxis, np.ones(range_num)]).T

    m, c = lin.lstsq(A, data)[0]

    # m, c = 20.0, 5
    print ( m, c)

    _x = m * np.array(xaxis) + c

    plt.plot(xaxis, data , 'o', label = 'data', markersize=10)

    plt.plot(xaxis, _x , 'r' , label='line')

    plt.legend()
    plt.show()

    # plt.plot(xaxis, data)
    # plt.grid(True)
    # plt.show()

STEP = 30
RANGE_NUM = 9000

def get_linvalue(data, focus=0):

    values = []

    window_size = STEP

    xaxis = [x for x in range(window_size)]

    for i in range(RANGE_NUM - window_size) :
        value = data[i : window_size + i]

        value = value[:, focus]

        A = np.vstack([xaxis, np.ones(window_size)]).T

        m, c = lin.lstsq(A, value)[0]

        _x = m * np.array(xaxis) + c

        # print ( m, c )
        # print (_x)
        #
        # print (_)

        # values.append(_x)
        values.append(c)

    return values

def get_autocorrelation_dataframe(series):
    def r(h):
        return ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    n = len(series)
    data = np.asarray(series)

    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    x = np.arange(n) + 1
    y = lmap(r, x)

    df = pd.DataFrame(y, index=x)

    return df


def get_half_life(df):
    price = pd.Series(df)

    lagged_price = price.shift(1).fillna(method="bfill")
    delta = price - lagged_price
    beta = np.polyfit(lagged_price, delta, 1)[0]
    half_life = (-1*np.log(2)/beta)

    return half_life

def draw_line (data , ax) :

    size = RANGE_NUM // STEP

    values1 = get_linvalue(data, 0)


    _xaxis = [x for x in range(len(values1))]

    # values2 = get_linvalue(data, size, 1)

    lines1 = ax.plot(_xaxis, values1, 'r', color='red', label='average')

    # lines2 = ax.plot(_xaxis, values2, 'r', color='blue', label='max')

    values3 = get_linvalue(data, 3)

    _xaxis = [x for x in range(len(values3))]

    lines3 = ax.plot(_xaxis, values3, 'r', color='green', label='moved average')

    ax.grid(True)

def main2 () :

    arr = load_avg_data()

    fig, axs = plt.subplots(1, 1)

    # data = arr[0:3000,0]

    values = []
    for i in range(100) :
        begin = STEP * i
        end = begin + STEP
        data = arr[begin:end, 3]

        result = get_half_life(data)

        values.append(result)

    plt.plot(values)

    # autocorrelation_plot(data , ax=axs)
    # axx = get_autocorrelation_dataframe(data)
    #
    # axx[0].plot(ax=axs)

    plt.show()

def main () :

    arr = load_avg_data()

    plot_count = 1

    fig = plt.figure(1)

    for i in range(plot_count) :
        begin = RANGE_NUM * i
        end = begin + RANGE_NUM

        data = arr[begin:end]

        ax = fig.add_subplot(plot_count, 1, i + 1)

        draw_line(data , ax)

    plt.legend()
    plt.show()

    # plt.plot(xaxis, data)
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__' :
    #main()
    # main2()
    lin1()