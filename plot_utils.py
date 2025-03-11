import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_parameter_trajectories(param_history,reference_iter=0, 
                                     log_scale_norm=True, figsize=(16, 10),
                                     title='Parameter Trajectories During Training'):
    """
    Visualize the trajectory of parameters over training iterations with both
    regular and normalized views in a single figure with two subplots.
    
    Parameters:
    -----------
    param_history : dict of lists of dicts
        Dictionary where keys are iteration numbers and values are lists of dictionaries,
        where each dictionary contains a single key:value pair.
        Format: {
            0: [{'param1': 0.1}, {'param2': 0.5}, ...],
            1: [{'param1': 0.09}, {'param2': 0.48}, ...],
            ...
        }
        
    reference_iter : int, optional
        The reference iteration for normalization (default: 0 for first iteration)
        
    log_scale_norm : bool, optional
        Whether to use log scale for y-axis in the first plot
        
    figsize : tuple, optional
        Figure size as (width, height)
        
    title : str, optional
        Main figure title
    
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axes objects
    """
    # Restructure the data into a more plot-friendly format
    iterations = sorted(param_history.keys())
    max_iter = max(iterations)
    target_iter = max_iter + 1  # Place target one step after the last iteration
    
    # Identify all unique parameter names across all iterations
    param_names = set()
    for iteration in iterations:
        for param_dict in param_history[iteration]:
            param_names.update(param_dict.keys())
            
    param_names = sorted(list(param_names))
    
    # Create a structured DataFrame with explicit conversion to Python floats
    data = []
    for iteration in iterations:
        row = {'iteration': iteration}
        
        # Extract parameter values for this iteration
        for param_dict in param_history[iteration]:
            for param_name, value in param_dict.items():
                # Handle JAX arrays by converting to NumPy float
                if hasattr(value, 'item'):
                    row[param_name] = float(value.item())
                elif isinstance(value, (list, tuple)) and len(value) == 1:
                    # Handle single-element lists/tuples
                    val = value[0]
                    if hasattr(val, 'item'):
                        row[param_name] = float(val.item())
                    else:
                        row[param_name] = float(val)
                else:
                    row[param_name] = float(value)
        
        data.append(row)

    df = pd.DataFrame(data)
    

    # Make sure reference_iter exists for normalization
    if reference_iter not in df['iteration'].values:
        reference_iter = df['iteration'].min()
    
    # Get parameter names from DataFrame
    actual_param_names = [col for col in df.columns if col != 'iteration']
    
    # Create normalized dataframe
    df_norm = df.copy()
    ref_idx = df[df['iteration'] == reference_iter].index[0]
    reference_values = df.iloc[ref_idx][actual_param_names].to_dict()
    
    for param in actual_param_names:
        # Convert to numpy array to ensure division works
        if param in reference_values:
            ref_val = float(reference_values[param])
            if ref_val != 0:  # Avoid division by zero
                df_norm[param] = df[param].values / ref_val
            else:
                # If reference is zero, set to NaN to avoid infinity
                df_norm[param] = np.nan
    
    # Define colors using a list of distinct colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c564b', 
    ]
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=False)
    #fig.suptitle(title, fontsize=16, y=0.98)
    
    # Plot 2: Normalized parameter values
    valid_values = []
    for i, param in enumerate(actual_param_names):
        if not df_norm[param].isna().all():  # Skip if all values are NaN
            color_idx = i % len(colors)
            
            param_values = df_norm[param].dropna().values
            valid_values.extend(param_values)
            
            ax.plot(df_norm['iteration'], df_norm[param], label=param, 
                    linewidth=1, color=colors[color_idx], alpha=0.7)
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Normalized to Iteration {reference_iter}')
    ax.set_xlim(df_norm['iteration'].min() , df_norm['iteration'].max())
    # set y-axis to log scale if requested
    if log_scale_norm:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a horizontal line at y=1 for the normalized plot to show the reference value
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.85)  # Make room for the title and legend
    
    return fig, ax