# Keras LAMB Optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
-----

Implementation of the LAMB optimizer from the paper [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes](https://arxiv.org/abs/1904.00962).

# Usage

```python

from keras_lamb import LAMBOptimizer

optimizer = LAMBOptimizer(0.001, weight_decay=0.2)
model.compile(optimizer, ...)
```

# Requirements
- Keras 2.2.4+
- Tensorflow 1.13+
