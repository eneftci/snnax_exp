from typing import Callable, Optional, Sequence, Union
import math

import equinox as eqx
import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp

from .stateful import StatefulLayer, StateShape, default_init_fn, StatefulOutput
from ...functional.surrogate import superspike_surrogate, SpikeFn
from chex import Array, PRNGKey

class TrainableArray(eqx.Module):
    data: Array
    requires_grad: bool = True

    def __init__(self, array, requires_grad=True):
        self.data = array
        self.requires_grad = requires_grad

class ComplexLIF(StatefulLayer):
    """
    Experimental Dudchenko & Fabre neuron model
    """
    threshold: float 
    stop_reset_grad: bool
    spike_fn: Callable
    b: TrainableArray
    log_log_alpha: TrainableArray
    log_dt: TrainableArray
    alpha_img: TrainableArray
    reset_val: float

    def __init__(self,
                shape: StateShape,
                log_alpha: float = .5 ,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                dt_min: float = 1e-2,
                dt_max: float = 7e-1,
                reset_val: float = .5,
                stop_reset_grad: bool = False,
                key: Optional[PRNGKey] = jax.random.PRNGKey(0), **kwargs) -> None:
        super().__init__()
        print(kwargs.keys(),  'unused kwargs')

        self.shape = shape
        # Fixed parameters
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.stop_reset_grad = stop_reset_grad
        self.reset_val = reset_val

        # Trainable parameters
        init_key, key = jax.random.split(key,2)
        self.b = TrainableArray(jax.random.uniform(shape=self.shape, key=init_key))
        self.log_log_alpha = TrainableArray(jnp.log(log_alpha * jnp.ones(self.shape)))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        init_key, key = jax.random.split(key,2)
        self.log_dt = TrainableArray(jax.random.uniform(shape=self.shape, key=init_key)*(math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.alpha_img =  TrainableArray(math.pi * jnp.ones(self.shape) )

    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey = None, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = jnp.zeros(shape, dtype=jnp.complex64, *args, **kwargs)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, st = state

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = jnp.exp((-jnp.exp(self.log_log_alpha.data)+1j*self.alpha_img.data)*jnp.exp(self.log_dt.data))
        # Loop over time axis

        # Compute membrane potential (LIF)
        if self.stop_reset_grad:
            mem_pot = alpha * (mem_pot - self.reset_val*self.threshold*jax.lax.stop_gradient(st)) + self.b.data * synaptic_input
        else:
            mem_pot = alpha * (mem_pot - self.reset_val*self.threshold*st) + self.b.data * synaptic_input 

        # Compute spikes with surrogate gradient
        spike_output = self.spike_fn(2*mem_pot.real - self.threshold)

        state = [mem_pot, spike_output]
        return [state, spike_output]

class AdaptiveLIF(ComplexLIF):

    def init_state(self, 
                   shape: Union[Sequence[int], int], 
                   key: PRNGKey = None, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        init_state_mem_pot = jnp.zeros(shape, dtype=jnp.float32, *args, **kwargs)
        init_state_adap = jnp.zeros(shape, dtype=jnp.float32, *args, **kwargs)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_adap, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, adap, st = state

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = jnp.exp((-jnp.exp(self.log_log_alpha.data)+1j*self.alpha_img.data)*jnp.exp(self.log_dt.data))
        # Loop over time axis

        ## Todo: add adaptation state
        # Compute membrane potential (LIF)
        if self.stop_reset_grad:
            mem_pot = alpha * (mem_pot - self.reset_val*self.threshold*jax.lax.stop_gradient(st)) + self.b.data * synaptic_input
        else:
            mem_pot = alpha * (mem_pot - self.reset_val*self.threshold*st) + self.b.data * synaptic_input 

        # Compute spikes with surrogate gradient
        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, adap, spike_output]
        return [state, spike_output]

