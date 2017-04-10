import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from collections import namedtuple
import tensorflow as tf

_data = [[[60,  80] , [120, 137]],
        [[145, 165] , [190, 212]],
        [[218, 240] , [298, 317]],
        [[328, 342] , [410, 442]],
        [[453, 467] , [610, 582]],
        [[588, 608] , [635, 658]],
        [[663, 683] , [725, 746]],
        [[757, 771] , [801, 879]],
        [[885, 904] , [940, 960]],
        [[967, 985] , [1030, 1138]],
        [[1147, 1162] , [1195, 1228]],
        [[1230, 1250] , [1280, 1318]],
        [[1330, 1343] , [1385, 1405]],
        [[1533, 1547] , [1572, 1588]],
        [[1881, 1902] , [1951, 1976]],
        [[1987, 2001] , [2028, 2047]],
        [[2053, 2071] , [2100, 2152]],
        [[2271, 2291] , [2314, 2334]],
        [[2343, 2364] , [2386, 2410]],
        [[2420, 2435] , [2467, 2502]],
        [[2780, 2794] , [2827, 2844]]]

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

STEP = 30
# RANGE_NUM = 9000

def get_lin(data , ax) :

    size = len(data)

    data = data[: , 3]

    xaxis = [x for x in range(size)]

    A = np.vstack([xaxis, np.ones(size)]).T

    m, c = lin.lstsq(A, data)[0]

    _x = m * np.array(xaxis) + c

    ax.plot(xaxis, data , 'o', label = 'data', markersize=2)

    ax.plot(xaxis, _x , 'r' , label='line')


def get_values(data, focus=0):
    pass



def main () :

    data = load_avg_data()

    fig = plt.figure(1)

    Lines = namedtuple('Lines', 'input_begin input_end output_begin output_end')

    plot_count = len(_data)

    for i in range(plot_count) :

        (_a, _b), (_c, _d) = np.array(_data[i]) * 30

        l = Lines(_a, _b , _c, _d)

        # print ( l)

        # value = data[l.input_begin:l.output_end]

        value = data[l.input_begin:l.input_end]
        #
        # value = data[l.output_begin:l.output_end]
        #
        #value = data[l.input_end:l.output_begin]

        ax = fig.add_subplot(plot_count, 1, i + 1)

        get_lin(value, ax)

        # _x, m, c = get_lin(value)
        #
        # ax.plot(_x, 'r', color='red', label='average')
        #
        # ax.set_ylabel('line %d'%i)

    #     draw_line(value , ax)
    #
    plt.legend()
    plt.show()

if __name__ == '__main__' :
    main()


    #
    # cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
    #
    # outputs, last_states = tf.nn.dynamic_rnn(
    #     cell=cell,
    #     dtype=tf.float64,
    #     sequence_length=X_lengths,
    #     inputs=X)
    #
    # result = tf.contrib.learn.run_n(
    #     {"outputs": outputs, "last_states": last_states},
    #     n=1,
    #     feed_dict=None)
    #