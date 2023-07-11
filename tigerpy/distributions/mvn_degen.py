"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from ..utils import (
    dot,
    quad_prod
)

from ..model.observation import Obs

from typing import Any
Array = Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

def _rank(eigenvalues: Array, tol: float = 1e-6) -> Array | float:
    """
    Computes the rank of a matrix based on the provided eigenvalues.
    The rank is taken to be the number of non-zero eigenvalues.

    Can handle batches.
    """
    mask = eigenvalues > tol
    rank = jnp.sum(mask, axis=-1)
    return rank

def _log_pdet(eigenvalues: Array, rank: Array | float | None = None,
             tol: float = 1e-6) -> Array | float:
    """
    Computes the log of the pseudo-determinant of a matrix based on the provided
    eigenvalues. If the rank is provided, it is used to select the non-zero eigenvalues.
    If the rank is not provided, it is computed by counting the non-zero eigenvalues. An
    eigenvalue is deemed to be non-zero if it is greater than the numerical tolerance
    ``tol``.

    Can handle batches.
    """

    if rank is None:
        mask = eigenvalues > tol
    else:
        max_index = eigenvalues.shape[-1] - rank

        def fn(i, x):
            return x.at[..., i].set(i >= max_index)

        mask = jax.lax.fori_loop(0, eigenvalues.shape[-1], fn, eigenvalues)

    selected = jnp.where(mask, eigenvalues, 1.0)
    log_pdet = jnp.sum(jnp.log(selected), axis=-1)
    return log_pdet

# (tfd.Distribution)
class MulitvariateNormalDegenerate:

    def __init__(self, loc: Array, scale: Array, rwk: int = 2,
                 rank: int = None,
                 log_pdet: float = None, tol: float = 1e-6):
        self.loc = loc
        self.scale = scale
        self.random_walk_order = rwk
        self.rank = rank
        self.log_pdet = log_pdet
        self.tol = tol
        self.dim = loc.shape[0]

        self.diff_mat = jnp.diff(jnp.eye(self.dim), n=self.random_walk_order, axis=0)
        self.prec = dot(self.diff_mat.T, self.diff_mat)
        self.eigenvals = jax.numpy.linalg.eigvalsh(self.prec)

        self.rank = _rank(eigenvalues=self.eigenvals, tol=self.tol) if self.rank is None else rank
        self.log_pdet = _log_pdet(eigenvalues=self.eigenvals, rank=self.rank) if self.log_pdet is None else log_pdet

    def log_prob(self, value):
        centered_value = value - self.loc
        logprob = - 0.5 * (self.rank * jnp.log(2*jnp.pi) - self.log_pdet) - 0.5 * quad_prod(self.prec, centered_value)
        return jnp.atleast_1d(logprob)
