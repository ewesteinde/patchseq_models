import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

import plotly.graph_objs as go
import plotly.offline as pyo

import jaxley as jx
from jaxley.modules import Branch, Cell, Compartment
from typing import Callable, List, Optional
import jax.numpy as jnp

def load_swc(input_file):
    """
    Load an SWC file into a pandas DataFrame.
    
    Parameters:
    -----------
    input_file : str
        Path to the input SWC file
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing SWC file data
    """
    # Define column names for SWC files
    columns = ['node_id', 'type', 'x', 'y', 'z', 'radius', 'parent']
    
    # Read the SWC file, skipping comment lines
    df = pd.read_csv(
        input_file, 
        delim_whitespace=True, 
        comment='#', 
        header=None, 
        names=columns
    )
    
    return df

def add_compartment_to_swc(df, comp_type=2, length=30, radius=0.5, num_points=30, num_comps=2, parent=None):
    """
    Add a new compartment (line of nodes) to the SWC DataFrame, creating a straight branch.
    The branch is created by adding `num_points` evenly spaced nodes along a line from the parent.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the SWC data
    comp_type : int
        The type of the new compartment (e.g., 1 for soma, 2 for dendrite, etc.)
    length : float
        The total distance (length) of the new branch
    radius : float
        The radius of each new node
    num_points : int, optional
        The number of points (nodes) to create along the branch. Default is 10.
    num_comps: int, optional
        The number of compartments to add. Default is 1.
    parent : int, optional
        The parent node ID for this new branch. Default is None, which sets parent to soma (node_id=1).
    
    Returns:
    --------
    df : pandas.DataFrame
        Updated DataFrame with the new compartments (nodes) added
    """
    for c in range(num_comps):

        # Default parent is soma (node_id=1)
        if c == 0 and parent is None:
            parent = 1  # Assuming the soma has node_id = 1
        elif c != 0:
            parent = parent_node_id

        
        # Get the parent (soma or other) position (coordinates) from the parent node
        parent_node = df[df['node_id'] == parent]
        if parent_node.empty:
            raise ValueError(f"Parent node with node_id {parent} does not exist in the SWC file.")
        
        parent_x, parent_y, parent_z = parent_node[['x', 'y', 'z']].values[0]
        
        # Calculate the direction vector for the branch (assuming we want to make it along the x-axis)
        # You can modify this to use any direction based on the parent node position.
        direction = np.array([1, 0, 0])  # Example: straight along the x-axis

        # Calculate the increment for each point along the line
        step_size = length / (num_points - 1) if num_points > 1 else length
        
        # Create a list to hold new rows (nodes) for the new compartment (branch)
        new_rows = []
        max_node_id = df['node_id'].max()
        # Generate the points for the branch and add them to the list
        for i in range(num_points):
            # Calculate the position of the new point along the direction
            progress = step_size * i
            new_x = parent_x + direction[0] * progress
            new_y = parent_y + direction[1] * progress
            new_z = parent_z + direction[2] * progress
            
            # Set the new node_id (incrementing from the max node_id)
            new_node_id = max_node_id + 1 #if not df.empty else 1
            max_node_id = new_node_id
            
            # Set the parent for the new node (parent is the previous node)
            new_parent = parent if i == 0 else parent_node_id
            
            # Append the new node information
            new_row = {
                'node_id': new_node_id,
                'type': comp_type,
                'x': new_x,
                'y': new_y,
                'z': new_z,
                'radius': radius,
                'parent': new_parent
            }
            new_rows.append(new_row)
            
            # Update the parent for the next iteration
            parent_node_id = new_node_id
        
        # Create a DataFrame for the new nodes and append it to the original DataFrame
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df

def process_swc(input_file, remove_types=None, add_axon=False, output_file=None):
    """
    Load an SWC file, remove specified compartment types, and optionally save the modified file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input SWC file
    remove_types : list or int or None, optional
        Compartment type(s) to remove. Can be a single type or list of types.
        If None, no compartments are removed.
    add_axon : bool, optional
        whether to add synthetic axon compartment 
    output_file : str, optional
        Path to save the modified SWC file. If None, no file is saved.
    
    Returns:
    --------
    df_modified : pandas.DataFrame
        DataFrame with specified compartment types removed
    """
    # Load the SWC file
    df = load_swc(input_file)
    
    # Ensure remove_types is a list
    if remove_types is None:
        remove_types = []
    elif isinstance(remove_types, int):
        remove_types = [remove_types]
    
    # Remove specified compartment types
    df_modified = df[~df['type'].isin(remove_types)]
    
    # Adjust parent references for remaining nodes
    def adjust_parent_references(data):
        # Create a mapping of old node IDs to new node IDs
        id_mapping = {old: new for new, old in enumerate(data['node_id'], 1)}
        
        # Update parent references
        data['parent'] = data['parent'].map(lambda x: id_mapping.get(x, -1) if x in id_mapping else -1)
        data['node_id'] = list(range(1, len(data) + 1))
        
        return data
    
    df_modified = adjust_parent_references(df_modified)

    if add_axon:
        df_modified = add_compartment_to_swc(df_modified)
        
    
    # Save to file if output_file is provided
    if output_file:
        # Write SWC file
        with open(output_file, 'w') as f:
            f.write('# Generated by SWC processing script\n')
            df_modified.to_csv(
                f, 
                sep=' ', 
                header=False, 
                index=False, 
                columns=['node_id', 'type', 'x', 'y', 'z', 'radius', 'parent']
            )
    
    return df_modified

def visualize_swc(df, title=None):
    """
    Visualize the SWC morphology in 3D with dynamic color mapping.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SWC data
    title : str, optional
        Custom title for the plot
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a color palette based on unique compartment types
    unique_types = df['type'].unique()
    color_palette = list(mcolors.TABLEAU_COLORS.values())[:len(unique_types)]
    type_colors = dict(zip(unique_types, color_palette))
    
    # Plot each unique compartment type
    for comp_type in unique_types:
        type_data = df[df['type'] == comp_type]
        
        # Plot points
        ax.scatter(
            type_data['x'], 
            type_data['y'], 
            type_data['z'], 
            c=type_colors[comp_type],
            label=f'Type {comp_type}',
            alpha=0.7,
            s=30  # Increased point size
        )
        
        # Plot connections
        for _, row in type_data.iterrows():
            if row['parent'] != -1:
                parent = df[df['node_id'] == row['parent']].iloc[0]
                ax.plot(
                    [row['x'], parent['x']],
                    [row['y'], parent['y']],
                    [row['z'], parent['z']],
                    c=type_colors[comp_type],
                    linewidth=1.5
                )
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Neuron Morphology')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Adjust legend
    ax.legend(title='Compartment Types', loc='best', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()

def visualize_swc_interactive(df, title=None, output_file=None):
    """
    Create an interactive 3D visualization of the SWC morphology using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SWC data
    title : str, optional
        Custom title for the plot
    output_file : str, optional
        Path to save the interactive HTML plot
    
    Returns:
    --------
    fig : plotly.graph_objs._figure.Figure
        Interactive Plotly figure
    """
    # Define a color palette
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Get unique compartment types
    unique_types = df['type'].unique()
    
    # Create the figure
    fig = go.Figure()
    
    # Store node traces for line connections
    for comp_type in unique_types:
        type_data = df[df['type'] == comp_type]
        
        # Color for this compartment type
        color = color_palette[comp_type % len(color_palette)]
        
        # Scatter plot for nodes
        scatter = go.Scatter3d(
            x=type_data['x'],
            y=type_data['y'],
            z=type_data['z'],
            mode='markers',
            name=f'Type {comp_type}',
            marker=dict(
                size=2,
                color=color,
                opacity=0.8
            )
        )
        fig.add_trace(scatter)
        
        # Add line connections
        for _, row in type_data.iterrows():
            if row['parent'] != -1:
                parent = df[df['node_id'] == row['parent']].iloc[0]
                line = go.Scatter3d(
                    x=[row['x'], parent['x']],
                    y=[row['y'], parent['y']],
                    z=[row['z'], parent['z']],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                )
                fig.add_trace(line)
    
    # Customize layout
    fig.update_layout(
        title=title or 'Neuron Morphology',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend_title_text='Compartment Types',
        width=600,
        height=400
    )
    
    # Save to HTML if output file is specified
    if output_file:
        pyo.plot(fig, filename=output_file, auto_open=False)
    
    return fig

def print_compartment_types(df, custom_mapping = None):
    """
    Print the unique compartment types in the SWC file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SWC data
    """
    standard_mapping = {
        0: 'undefined',
        1: 'soma',
        2: 'axon',
        3: '(basal) dendrite',
        4: 'apical dendrite',
        5: 'custom',
    }

    if custom_mapping is not None:
        mapping = custom_mapping
    else:
        mapping = standard_mapping
    
    type_counts = df['type'].value_counts()
    print("Compartment Types and Their Counts:")
    for comp_type, count in type_counts.items():
        print(f"Type {comp_type}: {mapping.get(comp_type, 'unknown')} ({count} nodes)")
    #print(type_counts)


def build_radiuses_from_xyzr_varComps(
    radius_fns: List[Callable],
    branch_indices: List[int],
    min_radius: Optional[float],
    ncomp_list: List[int],
) -> jnp.ndarray:
    """Return the radiuses of branches given SWC file xyzr.

    Returns an array of shape `(num_branches, ncomp)`.

    Args:
        radius_fns: Functions which, given compartment locations return the radius.
        branch_indices: The indices of the branches for which to return the radiuses.
        min_radius: If passed, the radiuses are clipped to be at least as large.
        ncomp: The number of compartments that every branch is discretized into.
    """
    # Compartment locations are at the center of the internal nodes.
    radiuses = np.array([])
    for branch in branch_indices:
        ncomp = ncomp_list[branch]
        non_split = 1 / ncomp
        range_ = np.linspace(non_split / 2, 1 - non_split / 2, ncomp)
        radiuses = np.append(radiuses, radius_fns[branch](range_))
        if min_radius is None:
            assert np.all(
                radiuses > 0.0
            ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
        else:
            radiuses[radiuses < min_radius] = min_radius

    return radiuses


def read_swc_varComps(
    fname: str,
    lcomp: Optional[int] = None,
    max_branch_len: Optional[float] = None,
    min_radius: Optional[float] = None,
    axon_ncomp: Optional[int] = None,
    assign_groups: bool = True
) -> Cell:
    """Reads SWC file into a `Cell`.

    Jaxley assumes cylindrical compartments and therefore defines length and radius
    for every compartment. The surface area is then 2*pi*r*length. For branches
    consisting of a single traced point we assume for them to have area 4*pi*r*r.
    Therefore, in these cases, we set lenght=2*r.

    Args:
        fname: Path to the swc file.
        lcomp: The desired compartment length, used to set number of compartments per branch.
        max_branch_len: If a branch is longer than this value it is split into two
            branches.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: If True, then the identity of reconstructed points in the SWC
            file will be used to generate groups `undefined`, `soma`, `axon`, `basal`,
            `apical`, `custom`. See here:
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        axon_ncomp: If passed, the number of compartments for axon branches.

    Returns:
        A `Cell` object.
    """

    parents, pathlengths, radius_fns, types, coords_of_branches = jx.io.swc.swc_to_jaxley(
        fname, max_branch_len=max_branch_len, sort=True, num_lines=None
    )
    nbranches = len(parents)
    comp = Compartment()
    ncomp_list = []
    lcomp_array = np.array([])
    branch_list = []
    for b in range(nbranches):
        b_length = pathlengths[b]
        if axon_ncomp is not None and types[b] == 2:
            ncomps = axon_ncomp
        else:
            ncomps = 1 + 2 * int(b_length/(2*lcomp))
        ncomp_list.append(ncomps)
        lcomp_array = np.append(lcomp_array, np.repeat(b_length, ncomps) / ncomps)
        branch = Branch([comp for _ in range(ncomps)])
        branch_list.append(branch)

    cell = Cell(
            branch_list, parents=parents, xyzr=coords_of_branches
        )

    # Also save the radius generating functions in case users post-hoc modify the number
    # of compartments with `.set_ncomp()`.
    cell._radius_generating_fns = radius_fns
    cell.set("length", lcomp_array)

    radiuses_each = build_radiuses_from_xyzr_varComps(
            radius_fns,
            range(len(parents)),
            min_radius,
            ncomp_list,
        )
    cell.set("radius", radiuses_each)

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    ind_name_lookup = {
        0: "undefined",
        1: "soma",
        2: "axon",
        3: "basal",
        4: "apical",
        5: "custom",
    }

    types = np.asarray(types).astype(int)
    if assign_groups:
        for type_ind in np.unique(types):
            if type_ind < 5.5:
                name = ind_name_lookup[type_ind]
            else:
                name = f"custom{type_ind}"
            indices = np.where(types == type_ind)[0].tolist()
            if len(indices) > 0:
                cell.branch(indices).add_to_group(name)

    return cell 
