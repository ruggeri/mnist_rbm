from collections import namedtuple
import numpy as np

Samples = namedtuple(
    'Samples', [
        'hidden_probs_pos',
        'hidden_values_pos',
        'visible_values_neg',
        'hidden_probs_neg',
        'hidden_values_neg',
    ]
)

def sigmoid(energy):
    return 1 / (1 + np.exp(energy))

def produce_samples(session, rbm_graph, visible_values_pos):
    W, hidden_bias, visible_bias = session.run([
        rbm_graph.W,
        rbm_graph.hidden_bias,
        rbm_graph.visible_bias,
    ])

    hidden_probs_pos = sigmoid(
        visible_values_pos.dot(W) + hidden_bias
    )
    hidden_values_pos = np.random.binomial(
        1,
        hidden_probs_pos,
    )

    visible_probs_neg = sigmoid(
        hidden_values_pos.dot(W.T) + visible_bias
    )
    visible_values_neg = np.random.binomial(
        1,
        visible_probs_neg,
    )
    hidden_probs_neg = sigmoid(
        visible_values_neg.dot(W) + hidden_bias
    )
    hidden_values_neg = np.random.binomial(
        1,
        hidden_probs_neg,
    )

    return Samples(
        hidden_probs_pos = hidden_probs_pos,
        hidden_values_pos = hidden_values_pos,
        visible_values_neg = visible_values_neg,
        hidden_probs_neg = hidden_probs_neg,
        hidden_values_neg = hidden_values_neg,
    )
