"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from ..model.observation import Obs

from functools import cached_property
from typing import Any
Array = Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

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
    eigenvalues. If the rank is provided, it is used to select the non-zero/positive eigenvalues.
    If the rank is not provided, it is computed by counting the non-zero. An
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

class MultivariateNormalDegenerate(tfjd.Distribution):

    def __init__(self,
                 loc: Array,
                 scale: Array,
                 pen: Array,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True,
                 name: str = "MultivariateNormalDegenerate",
                 tol: float = 1e-6):

        parameters = dict(locals())

        self._tol = tol
        self._pen = pen
        # since batches are passed from a multivariate normal shape is (batch,event(=1 in the univariate case))
        if scale.ndim == 1:
            self._scale = scale
        elif scale.ndim == 2 and scale.shape[-1] == 1:
            self._scale = jnp.squeeze(scale, axis=-1)

        prec = pen / jnp.expand_dims(self._scale ** 2, axis=(-2, -1))
        loc = jnp.atleast_1d(loc)

        if not prec.shape[-2] == prec.shape[-1]:
            raise ValueError(
                "`prec` must be square (the last two dimensions must be equal)."
            )

        try:
            jnp.broadcast_shapes(prec.shape[-1], loc.shape[-1])
        except ValueError:
            raise ValueError(
                f"The event sizes of `prec` ({prec.shape[-1]}) and `loc` "
                f"({loc.shape[-1]}) cannot be broadcast together. If you "
                "are trying to use batches for `loc`, you may need to add a "
                "dimension for the event size."
            )

        prec_batches = jnp.shape(prec)[:-2]
        loc_batches = jnp.shape(loc)[:-1]
        self._broadcast_batch_shape = jnp.broadcast_shapes(prec_batches, loc_batches)
        nbatch = len(self.batch_shape)

        self._prec = jnp.expand_dims(prec, tuple(range(nbatch - len(prec_batches))))
        self._loc = jnp.expand_dims(loc, tuple(range(nbatch - len(loc_batches))))

        eigenvals = jnp.linalg.eigvalsh(pen)
        self._rank = _rank(eigenvalues=eigenvals, tol=tol)
        log_pdet_pen = _log_pdet(eigenvalues=eigenvals, rank=self._rank)
        self._log_pdet = log_pdet_pen - self._rank * jnp.log(self._scale ** 2)

        super().__init__(dtype=pen.dtype,
                         reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats,
                         parameters=parameters,
                         name=name)

    @cached_property
    def eigenvals(self) -> Array:
        """Eigenvalues of the distribution's precision matrices."""
        return jnp.linalg.eigvalsh(self._prec)

    @cached_property
    def rank(self) -> Array | float:
        """Ranks of the distribution's precision matrices."""
        eigenvals = self.eigenvals
        return _rank(eigenvals, tol=self._tol)

    @cached_property
    def log_pdet(self) -> Array | float:
        """Log-pseudo-determinants of the distribution's precision matrices."""
        eigenvals = self.eigenvals
        return _log_pdet(eigenvals, self._rank, tol=self._tol)

    @property
    def prec(self) -> Array:
        """Precision matrices."""
        return self._prec

    @property
    def loc(self) -> Array:
        """Locations."""
        return self._loc

    def log_prob(self, x: Array) -> Array | float:
        """
        Method to calculate the log-probability.
        """
        x_centerd = x - self._loc
        # necessary for correct broadcasting in the quadratic form
        x_centerd = jnp.expand_dims(x_centerd, axis=-2)
        x_centerd_T = jnp.swapaxes(x_centerd, -2, -1)

        prob1 = - jnp.squeeze(x_centerd @ self._prec @ x_centerd_T, axis=(-2, -1))
        prob2 = self._rank * jnp.log(2 * jnp.pi) - self._log_pdet

        return 0.5 * (prob1 - prob2)

    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._prec)[-1],))

    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._prec)[-1],), dtype=jnp.int32)

    def _batch_shape(self):
        return tf.TensorShape(self._broadcast_batch_shape)

    def _batch_shape_tensor(self):
        return jnp.array(self._broadcast_batch_shape, dtype=jnp.int32)
