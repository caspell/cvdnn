import functools
from utils import common
import numpy as np
import tensorflow as tf
from collections import namedtuple

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceLabelling:

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(
            self.target * tf.log(self.prediction), [1, 2])
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def read_dataset():

    datas = common.load_avg_data()

    _data = common._data

    Lines = namedtuple('Lines', 'input_begin input_end output_begin output_end')

    Data = namedtuple('Data', 'data target sample')

    plot_count = len(_data)

    _data = np.array(_data)

    max_range = np.max([ e - s for s , e in _data[:,0,:]])

    r , _ , _ = np.shape(_data)

    _values = np.zeros((r, max_range * 30, 2))
    _labels = np.zeros((r, 1, 1))

    for i in range(plot_count) :

        (_a, _b), (_c, _d) = np.array(_data[i]) * 30

        l = Lines(_a, _b , _c, _d)

        # print ( l)

        # value = data[l.input_begin:l.output_end]

        value = datas[l.input_begin:l.input_end, 0::3]

        _values[i][:l.input_end - l.input_begin] = value

        _labels[i] = np.array([1])

    sample = (lambda x : Data(_values[:x] , _labels[:x] , sample))

    train = Data(_values , _labels , sample)

    test = Data(_values[:5] , _labels[:5], sample)

    return train, test

def train_data (train, test) :

    _, length, image_size = train.data.shape

    num_classes = train.target.shape[2]


    data = tf.placeholder(tf.float32, [None, length, image_size], name='data')

    target = tf.placeholder(tf.float32, [None, length, num_classes], name='target')

    dropout = tf.placeholder(tf.float32, name='dropout')

    model = SequenceLabelling(data, target, dropout)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for epoch in range(10):
        for _ in range(2):
            batch = train.sample(10)
            sess.run(model.optimize, {
                data: batch.data, target: batch.target, dropout: 0.5})

        error = sess.run(model.error, {
            data: test.data, target: test.target, dropout: 1})

        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))

if __name__ == '__main__':

    train, test = read_dataset()

    # print (train)

    train_data(train, test)
