# Ideas for the Future of Machine Learning

All work is original and done by Martin Loncaric.

## Iterative Neural Networks

Existing neural networks typically use lower-level layers to inform higher-level layers.
While this can work with copious data and training time, I have 3 intuitive reasons why this isn't ideal:
1. Our human understanding of low-level facts it also informed by high-level information.
For example, a human seeing a red apple in blue light can (accounting for the high-level fact that it is an apple) infer the low-level fact that it is actually red.
By using high-level layers to inform low-level layers as well, we should be able to train more accurate models
2. It feels wasteful to use each model weight just once in a trivial way (linear layers) during computation.
We should be able to train models faster (via a greater signal:noise ratio) if we use each weight multiple times (iteratively) during computation.
3. For certain tasks, such as the computer vision task of identifying whether a shape contains a loop, relatively high-level information needs to propagate through the entire image.
The best hand-coded algorithm for such a task would look like a breadth-first or depth-first search, so non-iterative neural nets would be unlikely to succeed at this task in a single shot.

