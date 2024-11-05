from typing import Callable, Optional, Sequence, Union
import math
import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp
from equinox._module import field, Module
from typing import Any, Literal, Optional, TypeVar, Union
from jaxtyping import Array, PRNGKeyArray
import jax.random as jrandom

from chex import Array, PRNGKey

import equinox as eqx
import jax.numpy as jnp

class ComplexLinear(eqx.nn.Linear):
    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)
    
    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**
    
        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
    
        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.
    
        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = jnp.zeros(wshape, dtype=jnp.complex64)
        bshape = (out_features_,)
        self.bias = jnp.zeros(bshape, dtype=jnp.complex64) if use_bias else None
    
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
