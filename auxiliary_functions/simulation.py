"""
Functions for landslide simulation
"""
# %% Required components
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq

from landlab import imshowhs_grid

# %% Generate earthquakes
def generate_acceleration_grid(grid, horizontal_max, vertical_max,
                            distribution="uniform", center=None,
                            random_center=False, seed=None,
                            plot_grids=False):
    """
    Generate arrays of horizontal and vertical acceleration values for a landlab grid.
    
    Parameters:
    ----------
    grid : RasterModelGrid
        The landlab grid to generate acceleration values for
    max_horizontal : float
        Maximum horizontal acceleration value at the center
    max_vertical : float
        Maximum vertical acceleration value at the center
    distribution : str, optional
        Distribution shape: "uniform", "circular", "square", "diamond", or "exponential" (default: "uniform")
    center : tuple, optional
        (row, col) coordinates of the center point. If None and random_center is False, 
        the center of the grid is used.
    random_center : bool, optional
        If True, a random center point will be selected. This overrides the center parameter.
    seed : int, optional
        Random seed for reproducibility when using random_center
        
    Returns:
    -------
    tuple
        (horizontal_acceleration, vertical_acceleration) - Two numpy arrays with values at grid nodes
    """
    if distribution != "uniform":
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get grid dimensions
        rows, cols = grid.shape
        
        # Initialize arrays for node values
        num_nodes = grid.number_of_nodes
        horizontal_accel = np.zeros(num_nodes)
        vertical_accel = np.zeros(num_nodes)
        
        # Get x and y coordinates of all nodes
        node_x = grid.x_of_node
        node_y = grid.y_of_node
        
        # Create mask of valid (non-NaN) nodes
        valid_mask = ~np.isnan(node_x) & ~np.isnan(node_y)
        
        # Calculate center coordinates
        if random_center:
            # Get indices of valid nodes
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                # Fallback if no valid nodes (shouldn't happen)
                valid_indices = np.arange(num_nodes)
            
            # Select a random valid node as center
            center_node = np.random.choice(valid_indices)
            center_x = node_x[center_node]
            center_y = node_y[center_node]
            
            # Try to get row, col for reporting
            try:
                center_row, center_col = grid.node_row_col(center_node)
                print(f"Randomly selected center at row={center_row}, col={center_col} (node={center_node})")
            except:
                print(f"Randomly selected center at node={center_node}, x={center_x}, y={center_y}")
                
        elif center is None:
            # Use geometric center of valid nodes
            valid_x = node_x[valid_mask]
            valid_y = node_y[valid_mask]
            
            if len(valid_x) > 0:
                center_x = np.mean(valid_x)
                center_y = np.mean(valid_y)
            else:
                # Fallback to all nodes (unlikely scenario)
                center_x = np.mean(node_x[~np.isnan(node_x)])
                center_y = np.mean(node_y[~np.isnan(node_y)])
        else:
            # Center is provided as (row, col)
            center_row, center_col = center
            # Convert row, col to node ID
            center_row = min(max(0, center_row), rows - 1)
            center_col = min(max(0, center_col), cols - 1)
            center_node = grid.grid_coords_to_node_id(center_row, center_col)
            center_x = node_x[center_node]
            center_y = node_y[center_node]
        
        # Calculate normalization factors based on distribution
        # Use only valid nodes for distance calculations
        valid_x = node_x[valid_mask]
        valid_y = node_y[valid_mask]
        
        if distribution == "circular" or distribution == "exponential":
            # Euclidean distance
            distances = np.sqrt((valid_x - center_x)**2 + (valid_y - center_y)**2)
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        elif distribution == "square":
            # Chebyshev distance (max of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            max_distance = max(np.max(x_distances) if len(x_distances) > 0 else 1.0, 
                            np.max(y_distances) if len(y_distances) > 0 else 1.0)
        elif distribution == "diamond":
            # Manhattan distance (sum of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            distances = x_distances + y_distances
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        # Ensure max_distance is not zero
        max_distance = max(max_distance, 1e-10)
        
        # Calculate acceleration values for each node
        for i in range(num_nodes):
            # Skip nodes with NaN coordinates
            if np.isnan(node_x[i]) or np.isnan(node_y[i]):
                horizontal_accel[i] = 0.0
                vertical_accel[i] = 0.0
                continue
            
            # Calculate distance from center
            dx = node_x[i] - center_x
            dy = node_y[i] - center_y
            
            if distribution == "circular":
                distance = np.sqrt(dx**2 + dy**2)
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "square":
                distance = max(abs(dx), abs(dy))
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "diamond":
                distance = abs(dx) + abs(dy)  # Manhattan distance
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "exponential":
                distance = np.sqrt(dx**2 + dy**2)
                # Scale factor for exponential decay
                decay_factor = 3.0  # Adjust for faster/slower decay
                factor = np.exp(-distance / (max_distance / decay_factor))
            
            # Set the acceleration values
            horizontal_accel[i] = horizontal_max * factor
            vertical_accel[i] = vertical_max * factor
    else:
        horizontal_accel = np.ones_like(grid.at_node["topographic__elevation"])
        vertical_accel = np.ones_like(grid.at_node["topographic__elevation"])
        
        horizontal_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan
        vertical_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan
        
        horizontal_accel[grid.core_nodes] *= horizontal_max
        vertical_accel[grid.core_nodes] *= vertical_max
    
    if plot_grids:
        # Plot PGA arrays
        plt.figure(layout='constrained')
        plt.subplot(1,2,1)
        imshowhs_grid(grid, "topographic__elevation",
                      plot_type='Drape1', drape1=horizontal_accel,
                      cmap='Reds', allow_colorbar=True, ticks_km=True)
        
        plt.subplot(1,2,2)
        imshowhs_grid(grid, "topographic__elevation",
                      plot_type='Drape1', drape1=vertical_accel,
                      cmap='Reds', allow_colorbar=True, ticks_km=True)
        
        plt.suptitle('Earthquake PGA (in multiples of g)')
        plt.show()
    
    print(f"{distribution} horizontal and vertical PGA arrays generated")
    return horizontal_accel, vertical_accel

# %% Trace landslide paths
def trace_paths_landslides(grid, starting_nodes, newmark_distances,
                           check_sum=False, print_path_details=False):
    """
    Calculates all downslope paths taken from a set of starting nodes constrained 
    by the newmark distance of each node and assigns proportions for each path

    Parameters
    ----------
    grid : Landlab RasterModelGrid
        DESCRIPTION.
    starting_nodes : numpy.ndarray
        1D array of node ids to trace paths for.
    max_distances : numpy.ndarray
        2D array of same size as the landlab grid containing the newmark distances for each node.
    check_sum : bool, optional
        Checks the sum of final proportions for all paths from a each node.
        The sum should equal to one. Prints node_ids that do not add up to one
        The default is False.
    print_path_details : bool, optional
        Lists each starting node_id and all subsequent node_ids and proportions 
        for the paths starting at that node.
        The default is False.

    Returns
    -------
    final_nodes: list of tuples
        Lists each path as a tuple of node_ids, starting with the initial node.
    final_proportions: list
        List containing the final proportions for each path defined in final_nodes
    path_details: dict of lists
        Dictionary of starting node_ids with a list of paths for each node
    
    if check_sum=True:
        proportion_not_zero: dict
            Dictionary listing each path where the final proportions do not 
            add up to one

    """
    receiver_nodes = grid.at_node['hill_flow__receiver_node']
    receiver_proportions = grid.at_node['hill_flow__receiver_proportions']
    boundary_nodes = set(grid.boundary_nodes)  # Convert to set for faster lookup
    
    final_nodes = []
    final_proportions = []
    path_details = {}  # To store details of each path for each starting node
    
    for node in tqdm(starting_nodes):
        newmark_distance = newmark_distances[node]
        stack = [(0.0, node, 1.0, [node])]  # (current_distance, current_node, current_proportion, path)
        path_details[node] = []  # Initialize path details for this starting node

        while stack:
            current_distance, current_node, current_proportion, path = heapq.heappop(stack)

            if current_distance >= newmark_distance:
                
                final_nodes.append((node, current_node))
                final_proportions.append(current_proportion)
                path_details[node].append((path, current_proportion))
                continue
            
            receivers = receiver_nodes[current_node]
            receiver_props = receiver_proportions[current_node]
            
            if receivers[0] == current_node:
                # print('a')
                break
                # print('b')
            else: 
                for receiver, prop in zip(receivers, receiver_props):
                        if receiver != -1:
                            new_distance = current_distance + calculate_node_distance(grid, current_node, receiver)
                            new_path = path + [receiver]
                            new_proportion = current_proportion * prop
        
                            # Check if the receiver node is a boundary node
                            if receiver in boundary_nodes or grid.at_node['topographic__elevation'][receiver] == np.nan:
                                final_nodes.append((node, current_node))
                                final_proportions.append(current_proportion)
                                path_details[node].append((path, current_proportion))
                            else:
                                heapq.heappush(stack, (new_distance, receiver, new_proportion, new_path))
    
    # Prints out all of the path node ids
    if print_path_details:
        print("Path Details:")
        for start_node in path_details:
            print(f"Starting Node: {start_node}")
            for path, proportion in path_details[start_node]:
                print(f"Path: {path}, Proportion: {proportion}")
    
    # Checks the sum total for all proportions of the paths
    if check_sum:                        
        proportion_sums = {}
        proportion_not_zero = []
        for (start_node, end_node), proportion in zip(final_nodes, final_proportions):
            if start_node not in proportion_sums:
                proportion_sums[start_node] = 0.0
            proportion_sums[start_node] += proportion
        for start_node, total_proportion in proportion_sums.items():
            if not np.isclose(total_proportion, 1.0):
                proportion_not_zero.append([start_node, total_proportion])
                # print(f"Proportions for node {start_node} do not sum to one: {total_proportion}")
                
        return final_nodes, final_proportions, path_details, proportion_not_zero
    
    else:
        return final_nodes, final_proportions, path_details

# %%% Distance between nodes
def calculate_node_distance(grid, node1, node2):
    """
    Calculates the rectilinear distance between two nodes in the grid

    Parameters
    ----------
    grid : Landlab RasterModelGrid
        DESCRIPTION.
    node1 : int
        Id of first node.
    node2 : int
        Id of second node.

    Returns
    -------
    float
        Rectilinear distance between two given nodes.

    """
    x1, y1 = grid.node_x[node1], grid.node_y[node1]
    x2, y2 = grid.node_x[node2], grid.node_y[node2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# %% Update soil depth
def update_soil_depth(paths, proportions, soil_depth):
    """
    Uses outputs from trace_paths_landslides to calculate change in soil depth

    Parameters
    ----------
    paths : list of tuples
        Lists each path as a tuple of node_ids, starting with the initial node
    proportions : list
        List containing the final proportions for each path defined in final_nodes
    soil_depth : numpy.ndarray
        1D array of floats representing the depth of soil in meters at each node

    Returns
    -------
    updated_soil_depth : numpy.ndarray
        1D array of floats representing the new soil depth in meters at each node
    soil_deposition : numpy.ndarray
        1D array of floats representing the amount of soil **deposited** at each node
    soil_erosion : numpy.ndarray
        1D array of floats representing the amount of soil **eroded** from each node

    """
    updated_soil_depth = soil_depth.copy()
    
    for path, proportion in tqdm(zip(paths, proportions)):
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            soil_movement = updated_soil_depth[start_node] * proportion
            if updated_soil_depth[start_node] - soil_movement < 0:
                soil_movement = updated_soil_depth[start_node]  # Move only the available soil
            # Soil erosion
            updated_soil_depth[start_node] -= soil_movement
            # Soil deposition
            updated_soil_depth[end_node] += soil_movement
    
    return updated_soil_depth