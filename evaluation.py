from sklearn.linear_model import LogisticRegression, SGDClassifier

def evaluate_lr(train_x, train_y, test_x, test_y):
    lr = SGDClassifier(
#        multi_class = 'multinomial',
#        solver = 'newton-cg',
    )
    lr.fit(
        np.around(train_x),
        np.argmax(labeled_train_y, axis = 1),
    )

    score = lr.score(
        np.around(test_x),
        np.argmax(mnist.test.labels, axis = 1),
    )
    return score

def sample_and_evaluate_lr(session, rbm_graph):
    train_samples = produce_samples(
        session,
        rbm_graph,
        labeled_train_x
    )
    test_samples = produce_samples(
        session,
        rbm_graph,
        dataset.test_x,
    )

    return evaluate_lr(
        train_x = train_samples.hidden_values_pos,
        train_y = labeled_train_y,
        test_x = test_samples.hidden_values_pos,
        test_y = dataset.test_y
    )
