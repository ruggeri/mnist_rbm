from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# TODO: Here you can decide how much labeled training data you have.
labeled_train_x, labeled_train_y = (
    np.around(mnist.train.images),
    np.argmax(mnist.train.labels, axis = 1),
)

test_x, test_y = (
    np.around(mnist.test.images),
    np.argmax(mnist.test.labels, axis = 1),
)

def next_batch(batch_size):
    visible_values_pos, _ = mnist.train.next_batch(batch_size)
    return np.around(visible_values_pos)
