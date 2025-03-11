import os
import subprocess
import numpy as np
from jaxley.optimize.transforms import Transform
from jax import Array
from jax.typing import ArrayLike
import jaxley as jx
import jax.numpy as jnp
from scipy import interpolate
import random
import matplotlib.pyplot as plt

def find_files(substring, directory):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matching_files.append(os.path.join(root, file))
    return matching_files


def call_load_data_in_allensdk(ID):
    """
    Calls load_data function from load_ephys.py in the allensdk environment
    """
    # Create wrapper script with the full function call including default parameters
    wrapper_code = f"""
import sys
from load_ephys import load_data

# Call the function with the ID and default parameters
load_data({ID})
"""
    
    # Get the directory where load_ephys.py is located
    script_dir = '/Users/elena.westeinde/Code/patch_seq'  # Directory containing load_ephys.py
    wrapper_path = os.path.join(script_dir, 'wrapper.py')
    
    # Write the wrapper script
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    # Command using conda run, changing to the correct directory first
    command = f"cd {script_dir} && conda run -n allensdk python wrapper.py"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("Command output:", result.stdout)
        return result
        
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e.stderr)
        raise
        
    finally:
        # Clean up wrapper file
        if os.path.exists(wrapper_path):
            os.remove(wrapper_path)
            
            
def uniform_log_scale(min_val, max_val, seed = 0):
    """
    Generates a random number uniformly sampled in the log space 
    between min_val and max_val.

    Args:
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.

    Returns:
        float: A random number sampled uniformly in log space.
    """
    np.random.seed(seed)
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_value = np.random.uniform(log_min, log_max)
    return np.exp(log_value)


def initialize_parameters(bounds, random_seed = 1):
    """
    Randomly initialize parameters within specified bounds.
    Values are created as JAX arrays with float64 dtype.
    
    Args:
        for_bounds (list): List of dictionaries containing parameter names and their (min, max) bounds
        pct (float): Percentage of the (lower) range to use for random initialization (default 0.5)
        
    Returns:
        list: List of dictionaries with randomly initialized parameter values as JAX arrays
    """
    params = []
    random.seed(random_seed)
    for bound_dict in bounds:
        for param_name, value in bound_dict.items():
            lower = value.lower
            # specific to jax sigmoid transform
            upper = value.upper #width + lower
            # Generate random value within bounds and convert to JAX array
            random_value = uniform_log_scale(lower, upper, seed=random_seed)
            #random_value = random.uniform(lower + low_range_pct ,lower + range_pct) always returned value near upper allowed range
            value_array = jnp.array([random_value], dtype=jnp.float64)
            params.append({param_name: value_array})
    
    return params


def resample_timeseries(time, data, new_dt=0.1):
    """
    Resample time series data to a new time step using cubic spline interpolation
    
    Args:
        time: Original time points
        data: Original data values
        new_dt: New time step (default 0.1)
    
    Returns:
        new_time: Resampled time points
        new_data: Resampled data values
    """
    # Create new time points
    new_time = np.arange(time[0], time[-1], new_dt)
    
    # Create cubic spline interpolator
    # s=0 means no smoothing, which minimizes artifacts
    f = interpolate.splrep(time, data, s=0)
    
    # Evaluate spline at new time points
    new_data = interpolate.splev(new_time, f)
    
    return new_time, new_data

def find_deflection(signal, threshold=0.5):
    # Get indices where signal crosses threshold
    crossings = np.where(np.abs(np.diff(signal > threshold)))[0]
    return crossings

####### DATA LOADING FUNCTIONS #######

def make_data_dict(sweep_ids, ephys_data, sweep_features, stim_dur = 1000, stim_time=None, pre_stim=None, post_stim=None, pre_buffer=0, post_buffer=0, dt = 0.1, new_dt=0.0001):
    
    """_summary_

    Args:
        sweep_ids (list): list of sweep ids
        ephys_data (dict): dictionary of ephys data arrays for each sweep
        sweep_info (dict): dictionary of feature info for each sweep
        stim_time (list len 2, optional): [start, end] specifying time window (s) to extract stimulus details. 
        pre_stim (float, optional): amount of simulation time before stimulus (ms). Defaults to None.
        post_stim (float, optional): amount of simulation time after stimulus (ms). Defaults to None.
        pre_buffer (float, optional): amount of time directly before stimulus (ms) to exclude from prestim window. Defaults to 0.
        post_buffer (float, optional): amount of time directly after stimulus (ms) to exclude from poststim window. Defaults to 0.
        dt (float, optional): simulation timestep in ms. Defaults to 0.1.
        new_dt (float, optional): data timestep in s. Defaults to 0.0001.

    Returns:
        _type_: _description_
    """
    # fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    data_dict = {}
    for id in sweep_ids:
        data_dict[id] = {'target': {}, 'input': {}}
        _, new_voltage = resample_timeseries(ephys_data['long_squares'][id]['time'], ephys_data['long_squares'][id]['voltage'], new_dt=new_dt)
        new_time, new_current = resample_timeseries(ephys_data['long_squares'][id]['time'], ephys_data['long_squares'][id]['current'], new_dt=new_dt)
        
        # axes[0].plot(new_time, new_voltage,linewidth=1)
        # axes[1].plot(new_time, new_current / 1e3,linewidth=1)
        
        if stim_time==None:
            sweep_info = sweep_features[int(id)]
            # stim_amp = sweep_info['stimulus_amplitude'] / 1e3 # pA --> nA
            #print("""TEMP FIX FOR STIM AMP, REPLACE WHEN INDEXING SWEEP FEATURES IS FIXED""")
            min_deflection = min(new_current[(new_time > 1) & (new_time < 2)])
            max_deflection = max(new_current[(new_time > 1) & (new_time < 2)])
            if max_deflection > abs(min_deflection):
                stim_amp = max_deflection / 1e3
            else:
                stim_amp = min_deflection / 1e3
            # stim_dur is incorrect in a subset of the data, need to manually calculate
            # for now can set to 1s
            #stim_dur = sweep_info['stimulus_duration'] * 1e3 # seconds --> ms
            
            stim_start = sweep_info['stimulus_start_time'] * 1e3 # seconds --> ms
            stim_end = stim_start + stim_dur # ms
        else:
            stim_idx = find_deflection(new_current[(new_time > stim_time[0]) & (new_time < stim_time[1])])
            stim_amp = max(new_current[(new_time > stim_time[0]) & (new_time < stim_time[1])]) / 1e3 # convert from pA to nA
            stim_dur = (stim_idx[1] - stim_idx[0]) * 0.0001 / 0.001 # convert from steps to seconds to ms
            stim_start = stim_idx[0] * 0.0001 / 0.001 # convert from steps to seconds to ms
            stim_end = stim_start + stim_dur # ms

        # Define stimulus parameters
        if pre_stim == None:
            window_start = stim_start  # ms pre stim onset
            pre_stim_length = (stim_start/dt)
        else:
            window_start = pre_stim
            pre_stim_length = (pre_stim/dt) - pre_buffer
        if post_stim == None:
            window_end = new_time[-1]-(stim_start + stim_dur)
            post_stim_length = (window_end/dt)
        else:
            window_end = post_stim  # ms post stim offset
            post_stim_length = (post_stim/dt) - post_buffer # idxs at 0.1ms/step
            
        i_delay = window_start  # stim starts at 1s, want to start rec at 700ms. 1000ms - 700ms = 300ms = delay
        i_dur = stim_dur  # ms
        # dt = 0.1  # time step in ms
        if window_end > 0:
            t_max = i_delay + i_dur + window_end  # ms
        else:
            t_max = i_delay + i_dur
        i_amp = stim_amp 

        stim_current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
        
        # find index of new_time that is closet to window_start / 1000
        start_time = ((stim_start- window_start) / 1000) # s
        start_idx = (np.abs(new_time - start_time)).argmin()
        end_time = ((new_time[start_idx] * 1e3) + t_max) / 1000 # s
        end_idx = (np.abs(new_time - end_time)).argmin()
        
        if post_stim_length > (end_time * 1000)/dt - stim_end/dt:
            print('Post stim length is longer than time after stim, reducing post stim length')
            print('Post stim length: ', post_stim_length, ' time after stim: ', (end_time * 1000)/dt - stim_end/dt)
            post_stim_length = (end_time * 1000)/dt - stim_end/dt
        
        data_dict[id]['target']['time'] = new_time[start_idx:end_idx]
        data_dict[id]['target']['voltage'] = new_voltage[start_idx:end_idx]
        data_dict[id]['target']['current'] = new_current[start_idx:end_idx]
        
        data_dict[id]['input']['i_delay'] = i_delay
        data_dict[id]['input']['i_dur'] = i_dur
        data_dict[id]['input']['dt'] = dt
        data_dict[id]['input']['t_max'] = t_max
        data_dict[id]['input']['i_amp'] = i_amp
        data_dict[id]['input']['current'] = stim_current
        data_dict[id]['input']['pre_stim_length'] = pre_stim_length
        data_dict[id]['input']['post_stim_length'] = post_stim_length
    # plt.show()
    return data_dict

def create_step_lr_scheduler(init_lr, reduced_lr, transition_step=20):
        """
        Creates a learning rate scheduler that drops the learning rate after a specified step.
        
        Args:
            init_lr: Initial learning rate
            reduced_lr: Reduced learning rate after transition_step
            transition_step: Step at which to reduce the learning rate (default: 50)
        
        Returns:
            An optax.Schedule object
        """
        def schedule_fn(step):
            return jnp.where(step < transition_step, init_lr, reduced_lr)
        
        return schedule_fn

##### CLASSES ##### 

class custom_SigmoidTransform(Transform):
    """Numerically stable sigmoid transformation that bijectively maps values to [lower, upper]."""
    
    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """Initialize transform with bounds.
        
        Args:
            lower: Lower bound of the target interval
            upper: Upper bound of the target interval
            
        Note: 
            Uses careful clipping and scaling to maintain numerical stability
            while preserving smoothness of the transformation.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.width = upper - lower
        
        # Constants for numerical stability
        self._eps = 1e-6  # Small epsilon for preventing division by zero
        self._clip_value = 20.0  # Smaller than float32 limit for safety
        
        # Precompute sigmoid bounds for the clipped domain
        self._sigmoid_lo = 1.0 / (1.0 + jnp.exp(self._clip_value))
        self._sigmoid_hi = 1.0 / (1.0 + jnp.exp(-self._clip_value))
        
        # Effective range after considering numerical bounds
        self._y_lo = self.lower + self.width * self._sigmoid_lo
        self._y_hi = self.lower + self.width * self._sigmoid_hi

    def forward(self, x: ArrayLike) -> Array:
        """Forward transformation from real numbers to [lower, upper].
        
        Args:
            x: Input values to transform
            
        Returns:
            Transformed values in [lower, upper]
        """
        # Clip inputs to prevent overflow in exp
        x_clipped = jnp.clip(x, -self._clip_value, self._clip_value)
        
        # Compute sigmoid with better numerical stability
        # Using log-space computation for large negative values
        z = jnp.exp(-jnp.abs(x_clipped))
        sigmoid = jnp.where(
            x_clipped >= 0,
            1.0 / (1.0 + z),
            z / (1.0 + z)
        )
        
        # Scale to target range with careful clipping
        y = self.lower + self.width * sigmoid
        return jnp.clip(y, self._y_lo, self._y_hi)

    def inverse(self, y: ArrayLike) -> Array:
        """Inverse transformation from [lower, upper] back to real numbers.
        
        Args:
            y: Input values from [lower, upper] to inverse transform
            
        Returns:
            Inverse transformed values
        """
        # Normalize to [0,1] interval with safety margins
        x = (y - self.lower) / self.width
        x = jnp.clip(x, self._sigmoid_lo + self._eps, self._sigmoid_hi - self._eps)
        
        # Inverse sigmoid with improved numerical stability
        # Using log-space computation to prevent overflow
        x_safe = jnp.clip(x, self._eps, 1.0 - self._eps)
        log_ratio = jnp.log(x_safe / (1.0 - x_safe))
        
        # Clip output to maintain consistency with forward transform
        return jnp.clip(log_ratio, -self._clip_value, self._clip_value)
    
class LogSpaceTransform(Transform):
    """Numerically stable logarithmic transformation that bijectively maps values to [lower, upper]."""
    
    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """Initialize transform with bounds.
        
        Args:
            lower: Lower bound of the target interval (must be positive)
            upper: Upper bound of the target interval (must be greater than lower)
            
        Note: 
            Uses log-space computations for numerical stability while handling
            a wide range of magnitudes. Particularly useful for parameters
            that span multiple orders of magnitude.
        """
        super().__init__()
        if not (jnp.all(lower > 0) and jnp.all(upper > lower)):
            raise ValueError("Lower bound must be positive and upper bound must exceed lower bound")
            
        self.lower = lower
        self.upper = upper
        
        # Precompute log-space constants
        self._log_lower = jnp.log(lower)
        self._log_upper = jnp.log(upper)
        self._log_range = self._log_upper - self._log_lower
        
        # Constants for numerical stability
        self._eps = 1e-6
        self._clip_value = 20.0
        
        # Precompute sigmoid bounds for the clipped domain
        self._sigmoid_lo = 1.0 / (1.0 + jnp.exp(self._clip_value))
        self._sigmoid_hi = 1.0 / (1.0 + jnp.exp(-self._clip_value))
        
        # Effective range after considering numerical bounds
        self._y_lo = jnp.exp(self._log_lower + self._log_range * self._sigmoid_lo)
        self._y_hi = jnp.exp(self._log_lower + self._log_range * self._sigmoid_hi)

    def forward(self, x: ArrayLike) -> Array:
        """Forward transformation from real numbers to [lower, upper].
        
        Args:
            x: Input values to transform
            
        Returns:
            Transformed values in [lower, upper]
        """
        # Clip inputs for numerical stability
        x_clipped = jnp.clip(x, -self._clip_value, self._clip_value)
        
        # Compute sigmoid with improved numerical stability
        z = jnp.exp(-jnp.abs(x_clipped))
        sigmoid = jnp.where(
            x_clipped >= 0,
            1.0 / (1.0 + z),
            z / (1.0 + z)
        )
        
        # Transform through log space
        log_y = self._log_lower + self._log_range * sigmoid
        y = jnp.exp(log_y)
        
        # Ensure output stays within bounds
        return jnp.clip(y, self._y_lo, self._y_hi)

    def inverse(self, y: ArrayLike) -> Array:
        """Inverse transformation from [lower, upper] back to real numbers.
        
        Args:
            y: Input values from [lower, upper] to inverse transform
            
        Returns:
            Inverse transformed values
        """
        # Clip inputs to valid range with safety margin
        y_safe = jnp.clip(y, self._y_lo + self._eps, self._y_hi - self._eps)
        
        # Transform to normalized space through log domain
        log_y = jnp.log(y_safe)
        x = (log_y - self._log_lower) / self._log_range
        
        # Compute inverse sigmoid with numerical stability
        x_safe = jnp.clip(x, self._sigmoid_lo + self._eps, self._sigmoid_hi - self._eps)
        log_ratio = jnp.log(x_safe / (1.0 - x_safe))
        
        # Clip output to maintain consistency with forward transform
        return jnp.clip(log_ratio, -self._clip_value, self._clip_value)

    def log_det_jacobian(self, x: ArrayLike) -> Array:
        """Compute log determinant of Jacobian for the forward transformation.
        
        Args:
            x: Input values
            
        Returns:
            Log determinant of Jacobian at input points
        """
        # Compute derivative of sigmoid
        x_clipped = jnp.clip(x, -self._clip_value, self._clip_value)
        sigmoid_deriv = jnp.exp(-x_clipped) / (1.0 + jnp.exp(-x_clipped))**2
        
        # Chain rule: d/dx[exp(log_lower + log_range * sigmoid(x))]
        log_det = jnp.log(self._log_range * sigmoid_deriv) + \
                  self._log_lower + \
                  self._log_range * self.forward(x)
                  
        return log_det

class Dataset:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.ids = list(data_dict.keys())
        self.current_idx = 0
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        current_id = self.ids[idx]
        #print(self.data_dict[current_id]['target']['current'].shape)
        return {
            'id': current_id,
            'target': {
                'time': self.data_dict[current_id]['target']['time'],
                'voltage': self.data_dict[current_id]['target']['voltage'],
                'current': self.data_dict[current_id]['target']['current']
            },
            'input': {
                'i_delay': self.data_dict[current_id]['input']['i_delay'],
                'i_dur': self.data_dict[current_id]['input']['i_dur'],
                'dt': self.data_dict[current_id]['input']['dt'],
                't_max': self.data_dict[current_id]['input']['t_max'],
                'i_amp': self.data_dict[current_id]['input']['i_amp'],
                'current': self.data_dict[current_id]['input']['current'],
                'pre_stim_length' : self.data_dict[current_id]['input']['pre_stim_length'],
                'post_stim_length' : self.data_dict[current_id]['input']['post_stim_length']
            }
        }

    def get_batch(self, batch_size=32, shuffle=True, seed = 1):
        if shuffle:
            np.random.seed(seed)
            batch_ids = np.random.choice(self.ids, size=batch_size, replace=False)
        else:
            start_idx = self.current_idx
            end_idx = min(start_idx + batch_size, len(self.ids))
            batch_ids = self.ids[start_idx:end_idx]
            self.current_idx = end_idx if end_idx < len(self.ids) else 0
        
        try:  
            batch_data = [self.__getitem__(self.ids.index(id_)) for id_ in batch_ids]
        except:
            print('Error in batch data retrieval, reshuffling')
            batch_ids = np.random.choice(self.ids, size=batch_size, replace=False)
            batch_data = [self.__getitem__(self.ids.index(id_)) for id_ in batch_ids]
            return None, None, None
        
        # Create arrays for current and voltage
        batch_amps = jnp.array([self.data_dict[id_]['input']['i_amp'] for id_ in batch_ids])
        batch_current = jnp.array([self.data_dict[id_]['input']['current'] for id_ in batch_ids])
        batch_target = jnp.array([self.data_dict[id_]['target']['voltage'] for id_ in batch_ids])

        return batch_data, batch_current, batch_target, batch_amps
    
    
    