import config
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

LABELED_TRAINING_SET_SIZE = int(
    config.LABELED_PERCENTAGE * mnist.train.images.shape[0]
)

print(f">>>Labeled Training Set Size: {LABELED_TRAINING_SET_SIZE}<<<")

labeled_train_x, labeled_train_y = mnist.train.next_batch(
    LABELED_TRAINING_SET_SIZE
)
labeled_train_x, labeled_train_y = (
    np.around(labeled_train_x),
    np.argmax(labeled_train_y, axis = 1),
)

test_x, test_y = (
    np.around(mnist.test.images),
    np.argmax(mnist.test.labels, axis = 1),
)

def next_batch(batch_size):
    visible_values_pos, _ = mnist.train.next_batch(batch_size)
    return np.around(visible_values_pos)
