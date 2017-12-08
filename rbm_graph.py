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
        tf.reduce_sum(visible_units_pos * visible_bias, axis = 1)
        +
        tf.reduce_sum(
            (
                tf.matmul(visible_units_pos, W)
                *
                hidden_units_pos
            ),
            axis = 1
        )
        +
        tf.reduce_sum(hidden_units_pos * hidden_bias, axis = 1)
    )

    # Negative side.
    visible_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_VISIBLE_UNITS]
    )
    hidden_units_neg = tf.placeholder(
        tf.float32, [None, config.NUM_HIDDEN_UNITS]
    )

    energy_neg = (
        tf.reduce_sum(visible_units_neg * visible_bias, axis = 1)
        +
        tf.reduce_sum(
            (
                tf.matmul(visible_units_neg, W)
                *
                hidden_units_neg
            ),
            axis = 1
        )
        +
        tf.reduce_sum(hidden_units_neg * hidden_bias, axis = 1)
    )

    energy_diff = tf.reduce_sum(energy_pos - energy_neg)

    train_op = tf.train.AdamOptimizer(
        learning_rate = config.LEARNING_RATE,
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
        train_op = train_op,
    )
