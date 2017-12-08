from sklearn.linear_model import LogisticRegression, SGDClassifier

import dataset
from sampling import produce_samples

def evaluate_lr(train_x, train_y, test_x, test_y):
    lr = SGDClassifier()
    lr.fit(train_x, train_y)
    accuracy = lr.score(test_x, test_y)
    return accuracy

def sample_and_evaluate_lr(session, rbm_graph):
    train_samples = produce_samples(
        session,
        rbm_graph,
        dataset.labeled_train_x
    )
    test_samples = produce_samples(
        session,
        rbm_graph,
        dataset.test_x,
    )

    # Using the probabilities for hidden units was very helpful.
    return evaluate_lr(
        train_x = train_samples.hidden_probs_pos,
        train_y = dataset.labeled_train_y,
        test_x = test_samples.hidden_probs_pos,
        test_y = dataset.test_y
    )
