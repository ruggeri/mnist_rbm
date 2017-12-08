## Binary RBM Features

There appears to be clear positive improvement when using the RBM to
do semi-supervised learning.

The logistic regression accuracy with a small 1% labeled set is
81.34%. This is using the regular 784 features.

If I use 100 RBM features, the accuracy is more like approximately
83-84%. There is some fluctuation, so it is a little hard to tell.

## Continuous RBM Features

I can also use the *probabilities* on the discrete RBM hidden
codes. This made a huge difference, and on the 1% sample I saw an
increase from 73% to 84% accuracy with 256 hidden units.

This shows that the RBM *must* be learning something valuable. Hooray!

## Continuous Hidden Features RBM Learning

I don't think I saw an improvement in RBM learning if I used the
probabilities for hidden variables rather than just do a sample.

I want to look more into this. It seems like what I really want to do
is to find the gradient of `p\tilde(v)`, not `p\tilde(v, h)`. But my
guess is that by using the probabilities of `h` in the relevant dot
products, I do indeed get this (up to a constant).

I was close to proving this but I didn't quite manage it. I must
return to this...
