import numpy as np
import matplotlib.pyplot as plt

def load_kitti_poses(file_path):
    """
    Load KITTI poses from a text file.
    
    Each line of the file should contain 12 floating point numbers representing 
    a 3x4 pose matrix. The 3x4 matrix is converted into a 4x4 homogeneous transformation
    by appending [0, 0, 0, 1] as the last row.
    
    Args:
        file_path (str): Path to the pose file.
        
    Returns:
        poses (list of np.ndarray): List of 4x4 transformation matrices.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # Split the line and convert each element to float.
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue  # Skip lines that don't have 12 values.
            # Reshape into a 3x4 matrix.
            pose_3x4 = np.array(vals).reshape(3, 4)
            # Create a 4x4 homogeneous transformation matrix.
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose_3x4
            poses.append(pose_4x4)
    return poses

def display_kitti_trajectory(ax, poses):
    """
    Plot a top–down view of the camera trajectory using translation components 
    from a list of 4x4 pose matrices.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis on which to plot the trajectory.
        poses (list of np.ndarray): List of 4x4 transformation matrices.
    """
    # Extract the translation vector (x, y, z) from each pose.
    trajectory = [pose[:3, 3] for pose in poses]
    traj_array = np.array(trajectory)
    
    ax.clear()
    if traj_array.shape[0] > 0:
        # Plot the trajectory line using the x and z coordinates.
        ax.plot(traj_array[:, 0], traj_array[:, 2], 'b-', linewidth=2, label="Camera Trajectory")
        # Plot individual points.
        ax.scatter(traj_array[:, 0], traj_array[:, 2], c='red')
    
    ax.set_title("Top-Down Camera Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    plt.draw()

# Example usage:
if __name__ == '__main__':
    # Path to the KITTI pose file. Adjust according to your dataset location.
    pose_file = "00/00.txt"  
    poses = load_kitti_poses(pose_file)
    
    # Create a Matplotlib figure and axis.
    fig, ax = plt.subplots()
    
    # Display the trajectory.
    display_kitti_trajectory(ax, poses)
    
    # Show the plot. You can also update it dynamically if needed.
    plt.show()
