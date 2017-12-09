The first major win was to use the probability vectors for the hidden
units as the representation the SVM is learned on.

With 100 units, the RBM features had an accuracy of about 90-91% with
an SVM versus 87-88% for just SVM. So clearly it learned and did
something positive. This is when training the SVM on the entire
dataset. If I restricted the labeled data to train on, I observed the
RBM added more benefit.

In terms of estimating the change in free energy, you can approach it
two ways:

(1) Approximate F(v_pos) by using a sampled hidden vector. Hinton says
    in the negative phase you might as well use the hidden
    probabilities to estimate F(v_neg). Hinton says don't use the
    hidden probabilities for the positive phase. (See below: he
    doesn't exactly say this).
(2) Just exactly compute F(v_pos), F(v_neg), disregarding the hidden
    vectors you generated.

Number two is what I do, and it is exactly what is suggested by
Murphy's Machine Learning book algorithm 27.3.

Note that the gradient of `F(v_pos) - F(v_neg)` is equal to the outer
product of `v_pos` and `E(h|v)` minus `v_neg` and `E(h|v_neg)`. This
is the contrastive divergence gradient. I misunderstood CD to be
proposing the first approach.

As a practical matter, on the small dataset I am working with, both
approaches seem to work fine. It does feel like approach #1 is
probably a consistent, if high variance, way of doing things. But I
haven't proven that...

Thank god this nightmare is over!
