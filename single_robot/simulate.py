import torch
from model import device
from hyperparam import MAX_DETECTION_DIST

def simulate_robot_step(
    location, direction, action, elapsed_time, map_size, binary_map, randomness_std=0.1
):
    """
    Simulate the robot's next state while considering obstacles in the binary map (CUDA-enabled).

    Args:
        location (tuple): Current (x, y) coordinates of the robot on the map.
        direction (float): Current direction of the robot in radians.
        action (tuple): (linear_velocity, angular_velocity) from the model.
        elapsed_time (float): Time elapsed since the last step (in seconds).
        map_size (tuple): (width, height) of the map.
        binary_map (torch.Tensor): Binary map where 1 indicates obstacles and 0 indicates free space.
        randomness_std (float): Standard deviation for Gaussian noise.
        device (str): Device to run the simulation on ("cuda" or "cpu").

    Returns:
        tuple: Next (location, direction) of the robot.
        bool: Whether the robot collided with an obstacle.
    """
    # Move inputs to device
    binary_map = binary_map.to(device)
    x, y = torch.tensor(location, device=device, dtype=torch.float32)
    linear_velocity, angular_velocity = action
    width, height = map_size

    # Add Gaussian noise to velocities
    noisy_linear_velocity = linear_velocity + torch.normal(
        mean=0, std=randomness_std * abs(linear_velocity), size=(1,), device=device
    )
    noisy_angular_velocity = angular_velocity + torch.normal(
        mean=0, std=randomness_std * abs(angular_velocity), size=(1,), device=device
    )

    # Update direction (radians)
    next_direction = direction + noisy_angular_velocity * elapsed_time
    next_direction %= 2 * torch.pi  # Ensure direction is within [0, 2π]

    # Update location based on noisy linear velocity and elapsed time
    next_x = x + noisy_linear_velocity * elapsed_time * torch.cos(next_direction)
    next_y = y + noisy_linear_velocity * elapsed_time * torch.sin(next_direction)

    # Ensure the robot stays within map boundaries
    next_x = torch.clamp(next_x, 0, width - 1)
    next_y = torch.clamp(next_y, 0, height - 1)

    # Check if the next location collides with an obstacle
    next_i, next_j = int(next_x.item()), int(next_y.item())  # Convert to map indices
    if binary_map[next_i, next_j] == 1:  # Obstacle detected
        return (location, direction, True)

    # No collision; update the robot's state
    return (next_x.item(), next_y.item()), next_direction.item(), False

def update_maps(
    frontier_map,
    robot_obstacle_map,
    location,
    direction,
    ground_truth_obstacle_map,
    max_detection_dist,
    max_detection_angle,
):
    """
    Update the frontier map and the robot's obstacle map based on sensor readings using vectorized raycasting.

    Args:
        frontier_map (torch.Tensor): Frontier map (0 = unexplored, 1 = explored).
        robot_obstacle_map (torch.Tensor): Current obstacle map (0 = free, 1 = obstacle).
        location (tuple): Current (x, y) location of the robot.
        direction (float): Current direction of the robot in radians.
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map (1 = obstacle, 0 = free).
        max_detection_dist (float): Maximum detection range of the sensor.
        max_detection_angle (float): Maximum detection angle (radians).
        device (str): Device to run the simulation ("cuda" or "cpu").

    Returns:
        tuple: Updated (frontier_map, robot_obstacle_map).
    """
    # Move data to device
    frontier_map = frontier_map.to(device)
    robot_obstacle_map = robot_obstacle_map.to(device)
    ground_truth_obstacle_map = ground_truth_obstacle_map.to(device)

    # Dimensions of the maps
    height, width = frontier_map.shape

    # Robot location
    x_robot, y_robot = location
    x_robot, y_robot = float(x_robot), float(y_robot)  # ensure float

    # Create angles for rays
    num_rays = 360
    num_steps = 100
    ray_angles = torch.linspace(
        direction - max_detection_angle / 2,
        direction + max_detection_angle / 2,
        steps=num_rays,
        device=device
    )

    # Steps along each ray
    ray_steps = torch.linspace(0, max_detection_dist, steps=num_steps, device=device)

    # Compute all coordinates: shape [num_rays, num_steps]
    # ray_angles: [R], ray_steps: [S]
    # Use broadcasting: 
    # cos(ray_angles) -> [R], sin(ray_angles) -> [R]
    # ray_steps -> [S]
    # After broadcasting: X, Y -> [R, S]
    ray_angles = ray_angles.unsqueeze(1)  # [R, 1]
    X = x_robot + ray_steps * torch.cos(ray_angles)  # [R, S]
    Y = y_robot + ray_steps * torch.sin(ray_angles)  # [R, S]

    # Convert to integers
    X_int = X.long()
    Y_int = Y.long()

    # In-bound mask
    in_bounds = (X_int >= 0) & (X_int < height) & (Y_int >= 0) & (Y_int < width)

    # Get obstacle map values for all points (mask in-bounds first)
    # Initialize a tensor to indicate if it's obstacle or not (out-of-bounds treated as no obstacle)
    obstacle_map_vals = torch.zeros_like(X_int, dtype=ground_truth_obstacle_map.dtype)
    valid_coords_mask = in_bounds
    obstacle_map_vals[valid_coords_mask] = ground_truth_obstacle_map[X_int[valid_coords_mask], Y_int[valid_coords_mask]]

    # Determine where obstacles are encountered
    # obstacle_map_vals == 1 indicates obstacle
    obstacles = (obstacle_map_vals == 1)

    # For each ray, we want the first step where an obstacle appears
    # If no obstacle, we return -1 for that ray
    # We can use argmax trick:
    # Convert obstacles to float and find the first occurrence of 1
    # If no obstacle found, we set first_obstacle_step = -1
    has_obstacle = obstacles.any(dim=1)
    # argmax will give the index of the first maximum value (which should correspond to the first True)
    first_obstacle_step = obstacles.float().argmax(dim=1)
    # Where no obstacle is found, set to -1
    first_obstacle_step = torch.where(has_obstacle, first_obstacle_step, torch.full_like(first_obstacle_step, -1))

    # Now we need to update the maps based on these steps.
    # For rays with no obstacle (first_obstacle_step == -1):
    #   All visited cells (within in_bounds) are free.
    # For rays with obstacle:
    #   All cells before the obstacle step are free.
    #   The obstacle cell itself is an obstacle.

    # Create a step index tensor for comparisons
    step_indices = torch.arange(num_steps, device=device).unsqueeze(0)  # [1, S]

    # Mask for obstacle rays (first_obstacle_step >= 0)
    obstacle_rays = first_obstacle_step >= 0

    # For rays with obstacles:
    # visited steps are those up to and including the obstacle step
    visited_mask = (step_indices <= first_obstacle_step.unsqueeze(1))  # [R, S]
    # obstacle cell is exactly at first_obstacle_step
    obstacle_cell_mask = (step_indices == first_obstacle_step.unsqueeze(1)) & obstacle_rays.unsqueeze(1)

    # For rays without obstacles:
    # visited steps are all steps in-bounds
    # If first_obstacle_step = -1, we want visited_mask to include all steps (no obstacle found)
    # We can handle this by setting all visited steps where first_obstacle_step = -1:
    no_obstacle_rays = (first_obstacle_step == -1)
    visited_mask = torch.where(
        no_obstacle_rays.unsqueeze(1),
        torch.ones_like(visited_mask, dtype=torch.bool),
        visited_mask
    )

    # Combine visited_mask with in_bounds to ensure we only update for valid cells
    visited_mask = visited_mask & in_bounds

    # Free cells: visited cells not obstacles
    free_cells_mask = visited_mask & ~obstacle_cell_mask

    # Flatten masks for advanced indexing
    free_cells_mask_flat = free_cells_mask.flatten()
    obstacle_cell_mask_flat = obstacle_cell_mask.flatten()

    # Flatten coordinates
    X_flat = X_int.flatten()
    Y_flat = Y_int.flatten()

    frontier_map = frontier_map.clone()
    robot_obstacle_map = robot_obstacle_map.clone()

    # Update frontier_map and robot_obstacle_map
    # Set visited free cells
    frontier_map[X_flat[free_cells_mask_flat], Y_flat[free_cells_mask_flat]] = 1
    robot_obstacle_map[X_flat[free_cells_mask_flat], Y_flat[free_cells_mask_flat]] = 0

    # Set obstacle cells
    if obstacle_cell_mask_flat.any():
        frontier_map[X_flat[obstacle_cell_mask_flat], Y_flat[obstacle_cell_mask_flat]] = 1
        robot_obstacle_map[X_flat[obstacle_cell_mask_flat], Y_flat[obstacle_cell_mask_flat]] = 1

    return frontier_map, robot_obstacle_map
