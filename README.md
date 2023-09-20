# A Stochastic Variational Inference Approach for Semiparametric Distributional Regression

## Overview

Repository for my masters thesis that studies and implements stochastic variational inference for semiparametric distributional regression models. Variational inference is a method to approximate a posterior distribution using a simpler variational distribution. This inference approach allows to learn a posterior distribution with optimization instead of sampling. The thesis studies a "black-box" variational inference algorithm that can be applied to wide variety of models. The thesis aims to study properties of the algorithm for Bayesian GAMLSS models.

The directory tigerpy includes python code of the masters thesis that is loosely written in a package style. The python package is called tigerpy and allows for Bayesian inference in distributional regression models that can also contain smooth additive effects (via the B-spline basis). Flexible regression specifications are enabled by its directed graph structure. The package [`networkx`](https://networkx.org/documentation/stable/index.html#) is used for efficient directed graph construction. Moreover makes the package use of [`jax`](https://jax.readthedocs.io/en/latest/index.html) for high-performance array computing and automatic differentiation.

The master thesis is written in [`quarto`](https://quarto.org/) and is fully contained in the thesis directory. Child chapters contained in the chapters directory are all imported into the master file `thesis_paper.qmd`.

A simulation study is contained in the simulation directory. While the playground directory contains `.ipynb` files that apply the inference algorithm to different models. Additionally some figures for the thesis are generated in the notebook `plots_paper.ipynb`.

The variational inference algorithm is based on [Ranganath et al. 2014](https://proceedings.mlr.press/v33/ranganath14.pdf) and [Kucukelbir et al. 2016](https://arxiv.org/pdf/1603.00788.pdf). A well written general introduction into variational inference is provided by [Zhang et al. 2017](https://arxiv.org/pdf/1711.05597.pdf).

## Structure of the repo

```
.
├── README.md
├── playground
├── simulation
├── thesis
│   ├── bib
│   ├── chapters
│   └── tex
└── tigerpy
    ├── bbvi
    ├── distributions
    └── model
```

## Literature

Some literature references that might provide good background when studying the repo.

- [Advances in Variational Inference](https://arxiv.org/pdf/1711.05597.pdf)
- [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)
- [Black Box Variational Inference](https://proceedings.mlr.press/v33/ranganath14.pdf)
- [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf)
- [Liesel](https://github.com/liesel-devs/liesel)

## Dev-Notes

### `precommit`

Use the [pre-commit](https://pre-commit.com/) system library.

### Coding Style

Try to follow the [Google `Python` Style Guide](https://google.github.io/styleguide/pyguide.html).
