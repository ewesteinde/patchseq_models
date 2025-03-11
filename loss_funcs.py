import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import jaxley as jx
from jax import vmap
from functools import partial


def soft_spike_integrate(v, threshold = 2.0, smoothness = 5.0):

    derivative = jnp.gradient(v, axis=1)
    sigmoid = jax.nn.sigmoid(smoothness * (derivative - threshold))
    integral = trapezoid(sigmoid, dx=0.1, axis=1)
    
    return integral

def smooth_max(x, mask, temperature=10.0):
    """Differentiable maximum using softmax."""
    # Set masked values to large negative value
    masked_x = jnp.where(mask, x, -1e9)
    # Compute softmax-based maximum
    weights = jax.nn.softmax(masked_x / temperature, axis=1)
    return jnp.sum(weights * x, axis=1)

def smooth_min(x, mask, temperature=5.0):
    """Differentiable maximum using softmax."""
    # Set masked values to large negative value
    masked_x = jnp.where(mask, -x, 1e9)
    # Compute softmax-based maximum
    weights = jax.nn.softmax(masked_x / temperature, axis=1)
    return -jnp.sum(weights * -x, axis=1)

def regularizer(params, bounds, penalty_weight = 1):
    """
    Regularizer to enforce parameter bounds.
    
    Args:
        params (list): List of dictionaries containing parameter values
        bounds (list): List of dictionaries containing parameter bounds
        
    Returns:
        float: Regularization penalty
    """
    total_penalty = 0.0
    max_value = 1e6
    # Iterate over each parameter dictionary and its corresponding bounds
    for param_dict, bound_dict in zip(params, bounds):
        for param_name, value in param_dict.items():
            for i, d in enumerate(bounds):
                if param_name in d[0][0]:
                    idx = i
            lower = bounds[idx][0][1].lower
            upper = bounds[idx][0][1].upper
            lower = jnp.clip(lower, -max_value, max_value)
            upper = jnp.clip(upper, -max_value, max_value)

            # compute violations with gradual increase
            lower_violation = jnp.maximum(0, lower * 1.02 - value) # penalize when value hits 98% of lower bound
            upper_violation = jnp.maximum(0, value - upper * 0.98) # penalize when value hits 98% of upper bound
            
            # Square violations after clipping to prevent explosion
            penalty = penalty_weight * (
                jnp.sum(jnp.clip(lower_violation, 0, 100)**2) + 
                jnp.sum(jnp.clip(upper_violation, 0, 100)**2)
            )

            total_penalty += penalty
            # penalty += jnp.where(value < lower, jnp.abs(value - lower), 0.0)
            # penalty += jnp.where(value > upper, jnp.abs(value - upper), 0.0)
    
    return total_penalty

def simulate_step(cell, params, amp, i_delay, i_dur, dt, t_max):
    # currently only for step currents
    #i_amp = jnp.array([jnp.max(jnp.abs(current))])
    i_amp = amp #[jnp.max(jnp.abs(current))]
    currents = jx.step_current(i_delay=i_delay, i_dur=i_dur, i_amp =i_amp, delta_t=dt, t_max=t_max)
    # must use .data_stimulate not .stimulate when using vmapping
    current_stim = cell.branch(0).loc(0.0).data_stimulate(current = currents)
    return jx.integrate(cell, params=params, delta_t=dt, data_stimuli=current_stim, voltage_solver="jaxley.stone")
    
batched_sim_step = vmap(simulate_step, in_axes=(None, None, 0, None, None, None, None))

def loss_from_v_batched(batched_v, batched_bmtk, windows):
    # window0: stimulus window
    # window1: stimulus onset

    mask0 = jnp.array(windows[0], dtype=bool)
    mask1 = jnp.array(windows[1], dtype=bool)

    # batched_output shape = (batch_size, rec types, n_time_steps)
    v = batched_v[:,0,:]

    if v.shape[1] != batched_bmtk.shape[1]:
        diff = v.shape[1] - batched_bmtk.shape[1]
        v = jnp.where(diff > 0, v[:, diff:], v[:, :batched_bmtk.shape[1]])

    stim_v = jnp.where(mask0, v, 0)
    stim_batched_bmtk = jnp.where(mask0, batched_bmtk, 0)

    mean_v_stim = jnp.sum(stim_v, axis = 1)/jnp.sum(mask0)
    mean_batched_bmtk_stim = jnp.sum(stim_batched_bmtk, axis = 1)/jnp.sum(mask0)

    mean_error = jnp.mean(jnp.sqrt((mean_v_stim - mean_batched_bmtk_stim)**2 + 1e-8))

    # Smooth max/min error
    max_error = jnp.mean((smooth_max(v, mask0) - smooth_max(batched_bmtk, mask0))**2)
    min_error = jnp.mean((smooth_min(v, mask0) - smooth_min(batched_bmtk, mask0))**2)

    spike_int_model = soft_spike_integrate(v)
    spike_int_target = soft_spike_integrate(batched_bmtk)
    spike_int_diff = jnp.mean((spike_int_model - spike_int_target)**2)
    
    #start_MSE = jnp.mean(jnp.sqrt(jnp.sum((jnp.where(mask1, v, 0) - jnp.where(mask1, batched_bmtk, 0))**2, axis = 1)/ jnp.sum(mask1) + 1e-8))
    
    dvdt_model = jnp.gradient(v, 0.1, axis=1)
    dvdt_target = jnp.gradient(batched_bmtk, 0.1, axis=1)

    # voltage phase, each element is a tuple of (voltage, dvdt)
    phase_model = jnp.stack([v[:, :-1], dvdt_model[:, :-1]], axis=2)
    phase_target = jnp.stack([batched_bmtk[:, :-1], dvdt_target[:, :-1]], axis=2)
    
    # Mean over time points and phase dimensions, then average across batch     
    dvdt_loss = jnp.mean(jnp.sum((phase_model - phase_target)**2, axis=2))

    total_loss =  mean_error * 2 + spike_int_diff * 10 + dvdt_loss * 0.1 + max_error * 0.5 + min_error
    losses = {'spike_int_diff': spike_int_diff * 10, 'mean_error': mean_error * 2, 'dvdt_loss': dvdt_loss * 0.1, 'max_error': max_error * 0.5, 'min_error': min_error}
    return total_loss, losses

def loss_from_v_batched_passive(batched_v, batched_bmtk, windows):
    # window0: stimulus onset
    # window1: stimulus offset

    mask0 = jnp.array(windows[0], dtype=bool)
    mask1 = jnp.array(windows[1], dtype=bool)

    # batched_output shape = (batch_size, rec types, n_time_steps)
    v = batched_v[:,0,:]

    if v.shape[1] != batched_bmtk.shape[1]:
        diff = v.shape[1] - batched_bmtk.shape[1]
        v = jnp.where(diff > 0, v[:, diff:], v[:, :batched_bmtk.shape[1]])
    
    start_MSE = jnp.mean(jnp.sum((jnp.where(mask0, v, 0) - jnp.where(mask0, batched_bmtk, 0))**2, axis = 1)/ jnp.sum(mask0))
    end_MSE = jnp.mean(jnp.sum((jnp.where(mask1, v, 0) - jnp.where(mask1, batched_bmtk, 0))**2, axis = 1)/ jnp.sum(mask1))
    
    total_loss = start_MSE + end_MSE
    losses = {'start_MSE': start_MSE, 'end_MSE': end_MSE}
    return total_loss, losses


@partial(jax.jit, static_argnames=('cell', 'i_delay', 'i_dur', 'dt', 't_max', 'windows', 'bounds'))
def loss_fun_step(cell, params, target_v, amps, i_delay, i_dur, dt, t_max, windows, bounds):
    # batched_output shape = (batch_size, rec types, n_time_steps)
    jaxley_output = batched_sim_step(cell, params, amps, i_delay, i_dur, dt, t_max) 
    total_loss, losses = loss_from_v_batched(jaxley_output, target_v, windows)
    reg = regularizer(params, bounds) #define a regularizer to had a penality for specific needs if desired
    return 1.0 * total_loss + 1e-8 + reg, losses

@partial(jax.jit, static_argnames=('cell', 'i_delay', 'i_dur', 'dt', 't_max', 'windows', 'bounds'))
def loss_fun_step_passive(cell, params, target_v, amps, i_delay, i_dur, dt, t_max, windows, bounds):
    # batched_output shape = (batch_size, rec types, n_time_steps)
    jaxley_output = batched_sim_step(cell, params, amps, i_delay, i_dur, dt, t_max) 
    total_loss, losses = loss_from_v_batched_passive(jaxley_output, target_v, windows)
    reg = regularizer(params, bounds) #define a regularizer to had a penality for specific needs if desired
    return 1.0 * total_loss + 1e-8 + reg, losses

