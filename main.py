from collections import namedtuple
import numpy as np
import tensorflow as tf

import config
import dataset
from evaluation import evaluate_lr, sample_and_evaluate_lr
from rbm_graph import make_rbm_graph
from sampling import produce_samples

def train_batch(session, rbm_graph, learning_rate_value):
    visible_values_pos = dataset.next_batch(config.BATCH_SIZE)
    samples = produce_samples(
        session,
        rbm_graph,
        visible_values_pos,
    )

    # TODO:
    # (1) Originally I used the actual hidden samples and tried to
    #     minimize -log p\tilde(v, h),
    # (2) Then I used the hidden sample probabilities; I am not sure
    #     that made any difference, but I should test.
    # (3) Last (and presently) I totally ignore these values, and just
    #     minimize -log p\tilde(v) marginalizing over all h. Again,
    #     not sure if this is better, and doesn't seem to be CD...
    session.run(
        rbm_graph.train_op,
        feed_dict = {
            rbm_graph.visible_units_pos: visible_values_pos,
            rbm_graph.hidden_units_pos: samples.hidden_probs_pos,
            rbm_graph.visible_units_neg: samples.visible_values_neg,
            rbm_graph.hidden_units_neg: samples.hidden_probs_neg,
            rbm_graph.learning_rate: learning_rate_value
        }
    )

def run(session):
    # Calculating initial performance
    initial_score = evaluate_lr(
        train_x = dataset.labeled_train_x,
        train_y = dataset.labeled_train_y,
        test_x = dataset.test_x,
        test_y = dataset.test_y
    )
    print(f">>>Initial Small Labeled Score: {initial_score}<<<")

    # Build and initialize graph.
    rbm_graph = make_rbm_graph()
    session.run(
        tf.global_variables_initializer()
    )

    # Interestingly, with random initialization LR accuracy is pretty
    # okay!
    lr_accuracy = sample_and_evaluate_lr(session, rbm_graph)
    print(f"Batch: {0} | LR Accuracy: {lr_accuracy}")

    learning_rate_value = config.LEARNING_RATE
    for batch_idx in range(1, config.NUM_BATCHES):
        train_batch(session, rbm_graph, learning_rate_value)

        if batch_idx % config.BATCHES_PER_EVALUATION == 0:
            lr_accuracy = sample_and_evaluate_lr(session, rbm_graph)
            print(f"Batch: {batch_idx} | LR Accuracy: {lr_accuracy}")

            learning_rate_value *= config.LEARNING_RATE_DECAY

with tf.Session() as session:
    run(session)
