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

def make_rbm_graph():
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

    energy_pos = (
        -tf.reduce_sum(visible_units_pos * visible_bias, axis = 1)
        +
        tf.reduce_sum(
            tf.log_sigmoid(
                -hidden_bias
                -tf.matmul(visible_units_pos, W)
            )
        )
    )

    # Negative side.
    visible_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_VISIBLE_UNITS]
    )
    hidden_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_HIDDEN_UNITS]
    )

    energy_neg = (
        -tf.reduce_sum(visible_units_neg * visible_bias, axis = 1)
        +
        tf.reduce_sum(
            tf.log_sigmoid(
                -hidden_bias
                -tf.matmul(visible_units_neg, W)
            )
        )
    )

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
