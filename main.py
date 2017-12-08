from collections import namedtuple
import numpy as np
import tensorflow as tf

def train_batch(session, rbm_graph):
    visible_values_pos = dataset.next_batch(BATCH_SIZE)
    samples = produce_samples(
        session,
        rbm_graph,
        visible_values_pos,
    )

    session.run(
        rbm_graph.train_op,
        feed_dict = {
            rbm_graph.visible_units_pos: visible_values_pos,
            rbm_graph.hidden_units_pos: samples.hidden_values_pos,
            rbm_graph.visible_units_neg: samples.visible_values_neg,
            rbm_graph.hidden_units_neg: samples.hidden_values_neg,
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
    lr_accuracy = evaluate_lr(session, rbm_graph)
    print(f"Batch: {0} | LR Accuracy: {lr_accuracy}")

    for batch_idx in range(1, NUM_BATCHES):
        train_batch(session, rbm_graph)

        if batch_idx % 100 == 0:
            lr_accuracy = evaluate_lr(session, rbm_graph)
            print(f"Batch: {batch_idx} | LR Accuracy: {lr_accuracy}")

with tf.Session() as session:
    run(session)
