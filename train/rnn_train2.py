from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import rnn_utils as rutils
import numpy as np

dataset = rutils.read_data()

# Parameters
learning_rate = 0.001
training_iters = 2880 * 5
# training_iters = 100000
# training_iters = 2880
batch_size = 64
display_step = 16

# Network Parameters
n_input = 2 # MNIST data input (img shape: 28*28)
n_steps = 30 # timesteps

n_hidden = 128 # hidden layer num of features
n_classes = 4 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input], 'x')
y = tf.placeholder("float", [None, n_classes], 'y')

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

    #tf.nn.dropout(top_h, self.dropout_prob)

pred = RNN(x, weights, biases)

pred_argmax = tf.argmax(pred,1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(pred_argmax, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:

        batch_x, batch_y ,_ = dataset.train.next_batch(batch_size)

        # print('batch_x : ' , np.shape(batch_x))
        # print('batch_y : ' , np.shape(batch_y))

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    test_len = 128

    test_data = dataset.test.data[:test_len].reshape((-1, n_steps, n_input))
    test_label = dataset.test.labels[:test_len]

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    pred_target = rutils.get_scenes(batch_size, time_limit=60 * 60 * 6)

    l, b, f, v = np.shape(pred_target)

    print('pred_target', (l, b, f, v))

    total_pred = []

    for d in pred_target:
        pred_result , result = sess.run([pred, pred_argmax], feed_dict={x:d})
        # print ( result )
        # total_pred
        total_pred=np.append(total_pred, result)

    print (np.shape(total_pred))
    print ('---------------------------')
    print ( total_pred )

    data_path = '/home/mhkim/data/numpy/sampyo/avg_train_v2'

    with open(data_path, 'w') as fd :
        np.save(fd, total_pred)


