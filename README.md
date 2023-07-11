# Black Box Variational Inference with JAX

Tigerpy is a python package that allows for bayesian inference in GAMs using variational inference. Flexible regression specifications are enabled by its graph structure. The package employs [`networkx`](https://networkx.org/documentation/stable/index.html#) for efficient graph construction. Moreover the package makes use of [`jax`](https://jax.readthedocs.io/en/latest/index.html) for high-performance array computing. The variational inference algorithm is based on [Black Box Variational Inference](https://proceedings.mlr.press/v33/ranganath14.pdf). An general introduction to variational inference can be found [here](https://arxiv.org/pdf/1601.00670.pdf).

## ToDo

1. Implement a more efficient graph structure :white_check_mark:
2. Allow for early stopping and make use of minibatches
3. Implement Bayesian B-splines regression
4. Allow for broader classes of variational distributions

## Literature

- [Black Box Variational Inference](https://proceedings.mlr.press/v33/ranganath14.pdf)
- [Variational Inference](https://arxiv.org/pdf/1601.00670.pdf)
- [Liesel](https://github.com/liesel-devs/liesel)

## Dev-Notes

### Coding Style

Follow the [Google `Python` Style Guide](https://google.github.io/styleguide/pyguide.html).

### `precommit`

Use the [pre-commit](https://pre-commit.com/) system library.
