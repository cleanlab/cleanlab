# State-of-the-art (2020) for learning with noisy labels on CIFAR-10.

Checkout the repo: [cgnorthcutt/confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10) to see how `cleanlab` is used to achieve state-of-the-art for learning with noisy labels on CIFAR-10.

The main procedure is simple:
1. Use cleanlab to find the label errors in CIFAR-10.
2. Remove them.
3. Train on the remaining cleaned data using CoTeaching.
