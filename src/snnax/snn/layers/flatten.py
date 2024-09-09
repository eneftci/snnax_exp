from typing import Optional

import equinox as eqx
from chex import Array, PRNGKey


class Flatten(eqx.Module):
    """
    Simple module to flatten the output of a layer. The input has to be a numpy
    or jax.numpy array with at least one dimension.
    """
    def __call__(self, 
                x: Array, *, 
                key: Optional[PRNGKey] = None) -> Array:
        return x.flatten()

class Reshape(eqx.Module):
    """
    Simple module to flatten the output of a layer. The input has to be a numpy
    or jax.numpy array with at least one dimension.
    """
    shape: Array
    def __init__(self, shape: Array):
        self.shape = shape
        super().__init__()

    def __call__(self, 
                x: Array, *, 
                key: Optional[PRNGKey] = None) -> Array:
        return x.reshape(self.shape)

