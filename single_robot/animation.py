import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def viz_binarr(arr):
    '''
    arr is indexed like so: rr[x_ind, y_ind]
    '''
    fig, ax = plt.subplots()  # Create a figure and axes
    cax = ax.imshow(arr.T, cmap="gray", origin="lower")  # Display the array as a heatmap
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    return fig

def viz_one(
    frontier_map,
    robot_obstacle_map,
    ground_truth_obstacle_map,
    robot_position,
    robot_direction,
    max_detection_dist,
    max_detection_angle,
    save_path=None,
):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Frontier Map", "Robot Obstacle Map", "Ground Truth Obstacle Map"]

    # Initialize the maps and robot position plots
    images = []
    for ax, title in zip(axes, titles):
        img = ax.imshow(np.zeros_like(frontier_map.cpu()), cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        images.append(img)

        map_size = frontier_map.shape
        ax.set_xlim(0, map_size[0])  # width of the map (rows)
        ax.set_ylim(0, map_size[1])  # height of the map (columns)

    images[0] = 1 - frontier_map.T
    images[1] = 1 - robot_obstacle_map.T
    images[2] = 1 - ground_truth_obstacle_map.T

    # Get robot state for this frame
    robot_x, robot_y = robot_position
    robot_direction = robot_direction

    # Add robot position and detection field of view
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx], cmap="gray", origin="lower")
        # ax.plot(
        #     robot_x, robot_y, "ro", markersize=8, label="Robot Position", zorder=5
        # )
        ax.plot(
            robot_x, robot_y, "ro", markersize=8, zorder=5
        )

        # Detection field of view
        angles = np.linspace(
            robot_direction - max_detection_angle / 2,
            robot_direction + max_detection_angle / 2,
            100,
        )
        detection_x = robot_x + max_detection_dist * np.cos(angles)
        detection_y = robot_y + max_detection_dist * np.sin(angles)

        # Draw detection arc
        ax.plot(detection_x, detection_y, "r--")

        # Draw detection radius
        ax.plot([robot_x, detection_x[0]], [robot_y, detection_y[0]], "r--")
        ax.plot([robot_x, detection_x[-1]], [robot_y, detection_y[-1]], "r--")

    return fig

def animate_robot_progress(
    frontier_maps,
    robot_obstacle_maps,
    ground_truth_obstacle_map,
    robot_positions,
    robot_directions,
    max_detection_dist,
    max_detection_angle,
    save_path=None,
):
    """
    Create an animation to visualize the robot's progress over time.

    Args:
        frontier_maps (list): List of frontier maps (one per frame).
        robot_obstacle_maps (list): List of robot obstacle maps (one per frame).
        ground_truth_obstacle_map (np.ndarray): Ground truth obstacle map (0 = free, 1 = obstacle).
        robot_positions (list): List of (x, y) positions of the robot over time.
        robot_directions (list): List of directions of the robot (in radians) over time.
        max_detection_dist (float): Maximum detection range of the sensor.
        max_detection_angle (float): Maximum detection angle of the sensor (radians).
        save_path (str, optional): If provided, saves the animation to the specified path.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Frontier Map", "Robot Obstacle Map", "Ground Truth Obstacle Map"]

    # Initialize the maps and robot position plots
    images = []
    for ax, title in zip(axes, titles):
        img = ax.imshow(np.zeros_like(frontier_maps[0].T.cpu()), cmap="gray", origin="lower")
        images.append(img)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Function to update each frame
    def update_frame(frame):
        # Update maps for the current frame
        for ax in axes:
            ax.clear()
            map_size = frontier_maps[0].shape
            ax.set_xlim(0, map_size[0])  # width of the map (rows)
            ax.set_ylim(0, map_size[1])  # height of the map (cols)
        images[0] = 1 - frontier_maps[frame].T.cpu()  # Frontier map
        images[1] = 1 - robot_obstacle_maps[frame].T.cpu()  # Robot obstacle map
        images[2] = 1 - ground_truth_obstacle_map.T.cpu()  # Ground truth map

        # Get robot state for this frame
        robot_x, robot_y = robot_positions[frame]
        robot_direction = robot_directions[frame]

        # Add robot position and detection field of view
        for idx, ax in enumerate(axes):
            ax.imshow(images[idx], origin="lower")
            ax.plot(
                robot_x, robot_y, "ro", markersize=8, label="Robot Position", zorder=5
            )

            # Detection field of view
            angles = np.linspace(
                robot_direction - max_detection_angle / 2,
                robot_direction + max_detection_angle / 2,
                100,
            )
            detection_x = robot_x + max_detection_dist * np.cos(angles)
            detection_y = robot_y + max_detection_dist * np.sin(angles)

            # Draw detection arc
            ax.plot(detection_x, detection_y, "r--", label="Detection Range")

            # Draw detection radius
            ax.plot([robot_x, detection_x[0]], [robot_y, detection_y[0]], "r--")
            ax.plot([robot_x, detection_x[-1]], [robot_y, detection_y[-1]], "r--")

            ax.legend()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(robot_positions), interval=5, repeat=False
    )

    # Save the animation if save_path is provided

    # Show the animation
    # plt.show()
    # Save the animation if save_path is provided
    # To save the animation using Pillow as a gif
    if save_path:
        writer = animation.PillowWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(save_path, writer=writer)


if __name__ == "__main__":
    # Simulate robot progress
    steps = 50
    robot_positions = [(50 + i, 50 + i) for i in range(steps)]
    robot_directions = [np.pi / 4 for _ in range(steps)]  # Constant direction

    # Example maps
    map_size = (100, 100)
    frontier_maps = [
        np.random.rand(*map_size) > 0.5 for _ in range(steps)
    ]  # Simulated dynamic frontier maps
    robot_obstacle_maps = [
        np.zeros(map_size) for _ in range(steps)
    ]  # Simulated dynamic robot obstacle maps
    ground_truth_obstacle_map = np.zeros(map_size)
    ground_truth_obstacle_map[30:40, 30:40] = 1  # Add a square obstacle

    animate_robot_progress(
        frontier_maps,
        robot_obstacle_maps,
        ground_truth_obstacle_map,
        robot_positions,
        robot_directions,
        max_detection_dist=15,
        max_detection_angle=np.pi / 3,
    )
