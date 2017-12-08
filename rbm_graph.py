from collections import namedtuple
import tensorflow as tf

import config

RBMGraph = namedtuple(
    'RBMGraph', [
        'W',
        'visible_bias',
        'hidden_bias',
        'visible_units_pos',
        'hidden_units_pos',
        'energy_pos',
        'visible_units_neg',
        'hidden_units_neg',
        'energy_neg',
        'energy_diff',
        'learning_rate',
        'train_op',
    ]
)

def marginal_free_energy(visible_units, W, visible_bias, hidden_bias):
    # This approach calculates the *marginal* energy of the visible
    # units, over all possible configurations of the hidden units. You
    # don't need to explicitly sum over all `h`, which is nice,
    # because of the simple structure of RBMs.
    return (
        -tf.reduce_sum(visible_units * visible_bias, axis = 1)
        +
        tf.reduce_sum(
            tf.log_sigmoid(
                -hidden_bias
                -tf.matmul(visible_units, W)
            ),
            axis = 1
        )
    )

def energy(visible_units, hidden_units, W, visible_bias, hidden_bias):
    # This approach calculates the exact energy for an entire
    # configuration of the machine.
    return (
        -tf.reduce_sum(visible_units * visible_bias, axis = 1)
        -
        tf.reduce_sum(
            tf.matmul(visible_units, W)
            *
            hidden_units,
            axis = 1
        )
        -tf.reduce_sum(hidden_units * hidden_bias, axis = 1)
    )

def make_rbm_graph():
    # Parameters
    W = tf.Variable(
        tf.truncated_normal([
            config.NUM_VISIBLE_UNITS,
            config.NUM_HIDDEN_UNITS
        ]),
        name = 'W'
    )

    visible_bias = tf.Variable(
        tf.zeros(config.NUM_VISIBLE_UNITS), name = 'visible_bias'
    )
    hidden_bias = tf.Variable(
        tf.zeros(config.NUM_HIDDEN_UNITS), name = 'hidden_bias',
    )

    # Positive side.
    visible_units_pos = tf.placeholder(
        tf.float32, [None, config.NUM_VISIBLE_UNITS]
    )
    hidden_units_pos = tf.placeholder(
        tf.float32, [None, config.NUM_HIDDEN_UNITS]
    )

    energy_pos = marginal_free_energy(
        visible_units = visible_units_pos,
        W = W,
        visible_bias = visible_bias,
        hidden_bias = hidden_bias
    )

    # Negative side.
    visible_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_VISIBLE_UNITS]
    )
    hidden_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_HIDDEN_UNITS]
    )

    energy_pos = marginal_free_energy(
        visible_units = visible_units_neg,
        W = W,
        visible_bias = visible_bias,
        hidden_bias = hidden_bias
    )

    # This is our estimate of the gradient of p(v).
    energy_diff = tf.reduce_sum(energy_pos - energy_neg)

    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    train_op = tf.train.AdamOptimizer(
        learning_rate = learning_rate
    ).minimize(energy_diff)

    return RBMGraph(
        W = W,
        visible_bias = visible_bias,
        hidden_bias = hidden_bias,
        visible_units_pos = visible_units_pos,
        hidden_units_pos = hidden_units_pos,
        energy_pos = energy_pos,
        visible_units_neg = visible_units_neg,
        hidden_units_neg = hidden_units_neg,
        energy_neg = energy_neg,
        energy_diff = energy_diff,
        learning_rate = learning_rate,
        train_op = train_op,
    )
