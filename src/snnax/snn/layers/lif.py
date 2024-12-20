from typing import Callable, Optional, Sequence, Union

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp
import equinox as eqx

from .stateful import StatefulLayer, StateShape, default_init_fn, StatefulOutput
from ...functional.surrogate import superspike_surrogate, SpikeFn
from chex import Array, PRNGKey

class TrainableArray(eqx.Module):
    data: Array
    requires_grad: bool = True

    def __init__(self, array, requires_grad=True):
        self.data = array
        self.requires_grad = requires_grad

class SimpleLIF(StatefulLayer):
    """
    Simple implementation of a layer of leaky integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    Requires one decay constant to simulate membrane potential leak.
    
    Arguments:
        `decay_constants` (Array): Decay constant of the simple LIF neuron.
        `spike_fn` (Array): Spike treshold function with custom surrogate gradient.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Array): Reset value after a spike has been emitted.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                        through the refectory potential.
        `init_fn` (Callable): Function to initialize the initial state of the 
                    spiking neurons. Defaults to initialization with zeros 
                    if nothing else is provided.
        `shape` (StateShape): if given, the parameters will be expanded into vectors and initialized accordingly
        `key` (PRNGKey): used to initialize the parameters when shape is not None
    """
    decay_constants: Union[Sequence[float], Array] 
    threshold: Array
    spike_fn: SpikeFn
    stop_reset_grad: bool
    reset_val: Optional[Array]

    def __init__(self,
                decay_constants: Array,
                spike_fn: SpikeFn = superspike_surrogate(10.),
                threshold: Array = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None,
                **kwargs) -> None:

        super().__init__(init_fn, shape)
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:
        alpha = lax.clamp(0.5, self.decay_constants[0], 1.0)
        mem_pot, spike_output = state
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input
        spike_output = self.spike_fn(mem_pot-self.threshold)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_val = jnn.softplus(self.reset_val)
            reset_pot = reset_val * spike_output
            
        # Optionally stop gradient propagation through refectory potential       
        refectory_potential = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential
        
        state = [mem_pot,  spike_output]
        return [state, spike_output]


class LIF(StatefulLayer):
    """
    TODO improve docstring
    Implementation of a leaky integrate-and-fire neuron with
    synaptic currents. Requires two decay constants to describe
    decay of membrane potential and synaptic current.

    Arguments:

        `decay_constants` (Array): Decay constants for the LIF neuron.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Array): Reset value after a spike has been emitted. 
                        Defaults to None.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                            through the refectory potential.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
                    Defaults to initialization with zeros if 
                    nothing else is provided.
        `shape` (Sequence[int]): Shape of the neuron layer.
        `key` (PRNGKey): Random number generator key for initialization of parameters.
    """
    decay_constants: Array
    threshold: Array
    spike_fn: SpikeFn
    reset_val: Array
    stop_reset_grad: bool

    def __init__(self, 
                decay_constants: Union[Sequence[float], Array],
                spike_fn: SpikeFn = superspike_surrogate(10.),
                threshold: Array = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None) -> None:

        super().__init__(init_fn, shape)
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def init_state(self, 
                    shape: StateShape, 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        
        # The synaptic currents are initialized as zeros
        init_state_syn_curr = jnp.zeros(shape) 
        
         # The spiking outputs are initialized as zeros
        init_state_spike_output = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_syn_curr, init_state_spike_output]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array,
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, syn_curr, spike_output = state
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output 
        else:
            reset_pot = (mem_pot-self.reset_val)*spike_output 

        # Optionally stop gradient propagation through refectory potential       
        refectory_potential = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential

        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        beta  = clamp(0.5, self.decay_constants[1], 1.0)
        
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1.-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


class LIFSoftReset(LIF):
    """
    Similar to LIF but reset is additive (relative) rather than absolute:
    If the neurons spikes: 
    $V \rightarrow V_{reset}$
    where $V_{reset}$ is the parameter reset_val
    """
    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array,
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, syn_curr, spike_output = state
        
        if self.reset_val is None:
            reset_pot = spike_output 
        else:
            reset_pot = self.reset_val*spike_output

        # optionally stop gradient propagation through refectory potential       
        refr_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refr_pot

        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        beta  = clamp(0.5, self.decay_constants[1], 1.0)
 
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1.-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


class AdaptiveLIF(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented in https://neuronaldynamics.epfl.ch/online/Ch6.S1.html.
    
    Arguments:
        `decay_constants` (Array): Decay constants for the LIF neuron.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Array): Reset value after a spike has been emitted. 
                        Defaults to None.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                        through the refectory potential.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        `shape` (StateShape): Shape of the neuron layer.
        `key` (PRNGKey): Random number generator key for initialization of parameters.
    """
    decay_constants: Array
    threshold: Array
    ada_step_val: Array 
    ada_decay_constant: Array 
    ada_coupling_var: Array 
    stop_reset_grad: bool
    reset_val: Optional[Array]
    spike_fn: Callable

    def __init__(self,
                decay_constants: float,
                ada_decay_constant: float = [.8] ,
                ada_step_val: float = [1.0],
                ada_coupling_var: float = [.5],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None) -> None:
        super().__init__(init_fn)

        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val 
        self.stop_reset_grad = stop_reset_grad

        self.decay_constants = self.init_parameters(decay_constants, shape)
        self.ada_decay_constant = self.init_parameters(ada_decay_constant, shape)
        self.ada_step_val = self.init_parameters(ada_step_val, shape)
        self.ada_coupling_var = self.init_parameters(ada_coupling_var, shape)
       
    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_ada, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, ada_var, st = state

        alpha = clamp(0.5,self.decay_constants[0],1.)
        beta = clamp(0.5, self.ada_decay_constant[0], 1.) 
        a = clamp(-1.,self.ada_coupling_var[0], 1.)
        b = clamp(0.,self.ada_step_val[0], 2.)

        # Calculation of the membrane potential
        mem_pot = alpha*mem_pot + (1.-alpha)*(synaptic_input+ada_var)
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Calculation of the adaptive part of the dynamics
        ada_var_new = (1.-beta)*a * mem_pot \
                    + beta*ada_var - b*stop_gradient(spike_output)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val * spike_output
            
        # Optionally stop gradient propagation through refectory potential       
        refectory_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        state = [mem_pot, ada_var_new, spike_output]
        return [state, spike_output]

class AdLIF(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented Bittar and Garner
    
    Arguments:
        `decay_constants` (Array): Decay constants for the LIF neuron.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        `shape` (StateShape): Shape of the neuron layer.
        `key` (PRNGKey): Random number generator key for initialization of parameters.
    """
    alpha: Array
    beta: Array 
    a: Array 
    b: Array 
    threshold: Array
    spike_fn: Callable

    def __init__(self,
#                decay_constants: float,
#                ada_decay_constant: float = [.8] ,
#                ada_step_val: float = [1.0],
#                ada_coupling_var: float = [.5],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                init_fn: Optional[Callable] = None,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None, **kwargs) -> None:
        super().__init__(init_fn)
        print('AdLIF unused kwargs', kwargs.keys())

        self.threshold = threshold
        self.spike_fn = spike_fn
        init_key, key = jax.random.split(key,2)
        self.alpha = TrainableArray(jax.random.uniform(minval=jnp.exp(-1/5), maxval=jnp.exp(-1/25),shape=shape, key=init_key))
        init_key, key = jax.random.split(key,2)
        self.beta = TrainableArray(jax.random.uniform(minval=jnp.exp(-1/30), maxval=jnp.exp(-1/120),shape=shape,key=init_key))
        init_key, key = jax.random.split(key,2)
        self.a = TrainableArray( jax.random.uniform(minval=-1, maxval=1., shape=shape, key=init_key) )
        init_key, key = jax.random.split(key,2)
        self.b = TrainableArray( jax.random.uniform(minval=0, maxval=2., shape=shape, key=init_key) )
       
    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_ada, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        ut, wt, st = state

        alpha = clamp(jnp.exp(-1/5),self.alpha.data, jnp.exp(-1/25))
        beta = clamp(jnp.exp(-1/30), self.beta.data, jnp.exp(-1/120)) 
        a = clamp(-1.,self.a.data, 1.)
        b = clamp(0.,self.b.data, 2.)

        # Calculation of the adaptive part of the dynamics
        wt = a * ut  + beta*wt + b*st
        # Calculation of the membrane potential
        ut = alpha*(ut-st) + (1.-alpha)*(synaptic_input - wt)
        st = self.spike_fn(ut - self.threshold)

        # Optionally stop gradient propagation through refectory potential       
        state = [ut, wt, st]
        return [state, st]

class AdLIFfixtau(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented Bittar and Garner
    
    Arguments:
        `decay_constants` (Array): Decay constants for the LIF neuron.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        `shape` (StateShape): Shape of the neuron layer.
        `key` (PRNGKey): Random number generator key for initialization of parameters.
    """
    alpha: Array
    beta: Array 
    a: Array 
    b: Array 
    threshold: Array
    spike_fn: Callable

    def __init__(self,
                decay_constants: float = [.95],
                ada_decay_constant: float = [.9] ,
                ada_step_val: float = [1.0],
                ada_coupling_var: float = [.5],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                init_fn: Optional[Callable] = None,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None, **kwargs) -> None:
        super().__init__(init_fn)
        print('key', key)
        print('AdLIF unused kwargs', kwargs.keys())

        self.threshold = threshold
        self.spike_fn = spike_fn
        init_key, key = jax.random.split(key,2)
#        self.alpha = jax.random.uniform(minval=jnp.exp(-1/5), maxval=jnp.exp(-1/25),shape=shape, key=init_key)
#        init_key, key = jax.random.split(key,2)
#        self.beta = jax.random.uniform(minval=jnp.exp(-1/30), maxval=jnp.exp(-1/120),shape=shape,key=init_key)
#        init_key, key = jax.random.split(key,2)
#        self.a = jax.random.uniform(minval=-1, maxval=1., shape=shape, key=init_key) 
#        init_key, key = jax.random.split(key,2)
#        self.b = jax.random.uniform(minval=0, maxval=2., shape=shape, key=init_key) 
#
#        self.alpha = clamp(jnp.exp(-1/5),self.alpha, jnp.exp(-1/25))
#        self.beta = clamp(jnp.exp(-1/30), self.beta, jnp.exp(-1/120)) 
#        self.a = clamp(-1.,self.a, 1.)
#        self.b = clamp(0.,self.b, 2.)
        self.alpha = clamp(jnp.exp(-1/5), decay_constants[0], jnp.exp(-1/25))
        self.beta = clamp(jnp.exp(-1/30), ada_decay_constant[0], jnp.exp(-1/120)) 
        self.a = clamp(-1., ada_step_val[0], 1.)
        self.b = clamp(0., ada_coupling_var[0], 2.)

    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_ada, init_state_spikes]
       
    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        ut, wt, st = state

        # Calculation of the adaptive part of the dynamics
        # Calculation of the membrane potential
        wt = self.a * ut  + self.beta*wt + self.b*st
        ut = self.alpha*(ut-st) + (1.-self.alpha)*(synaptic_input - wt)
        st = self.spike_fn(ut - self.threshold)

        # Optionally stop gradient propagation through refectory potential       
        state = [ut, wt, st]
        return [state, st]

