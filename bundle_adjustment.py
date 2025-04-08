import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def rotate_point(angle_axis, point):
    """
    Rotate a 3D point using Rodrigues' rotation formula.
    
    Args:
        angle_axis (np.ndarray): (3, 1) axis-angle representation of rotation.
        point (np.ndarray): (3, 1) point to be rotated.
    
    Returns:
        np.ndarray: (3, 1) rotated point.
    """
    angle_axis = np.asarray(angle_axis, dtype=float).reshape(3, 1)
    point = np.asarray(point, dtype=float).reshape(3, 1)
    
    theta = np.linalg.norm(angle_axis)
    
    # If the rotation is too small, return the original point.
    if theta < 1e-8:
        return point
    
    # Normalize the rotation axis (k)
    k = angle_axis / theta
    
    # Compute the sine and cosine of the rotation angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Compute the cross and dot products
    k_cross_p = np.cross(k.flatten(), point.flatten()).reshape(3, 1)
    k_dot_p = np.dot(k.flatten(), point.flatten()).reshape(1, 1)
    
    # Rodrigues' rotation formula:
    # v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k dot v)*(1 - cos(theta))
    rotated_point = point * cos_theta + k_cross_p * sin_theta + k * (k_dot_p * (1 - cos_theta))
    return rotated_point

def project_point(point, camera_matrix):
    """
    Project a 3D point to 2D using the camera projection matrix.
    
    Args:
        point (np.ndarray): (3, 1) 3D point to be projected.
        camera_matrix (np.ndarray): (3, 4) camera projection matrix.
        
    Returns:
        np.ndarray: (2, 1) 2D projected point.
    """
    point = np.asarray(point, dtype=float).reshape(3, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 4)
    # Convert to homogeneous coordinates: [X, Y, Z, 1]
    point_homogeneous = np.vstack((point, [[1]]))
    projected_point = camera_matrix @ point_homogeneous
    projected_point /= projected_point[2]
    return projected_point[:2]

def ba_residuals(params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right):
    """
    Compute the reprojection residuals for bundle adjustment using projection matrices for left and right cameras.
    
    Args:
        params (np.ndarray): Combined parameter vector of shape 
                             (n_cameras * 6 + n_points * 3,). The first (n_cameras * 6)
                             entries represent camera poses (6 parameters each: 3 for
                             angle-axis rotation and 3 for translation), and the remaining 
                             entries represent the 3D landmarks (3 parameters each).
        n_cameras (int): Number of cameras.
        n_points (int): Number of points.
        camera_indices (np.ndarray): Array of shape (n_observations,) mapping each observation to a camera index.
        point_indices (np.ndarray): Array of shape (n_observations,) mapping each observation to a 3D point index.
        observations (np.ndarray): Array of shape (n_observations, 4) where each row is 
                                   [x_left, y_left, x_right, y_right] representing observed pixel coordinates.
        P_left (np.ndarray): (3, 4) left camera projection matrix.
        P_right (np.ndarray): (3, 4) right camera projection matrix.
        
    Returns:
        np.ndarray: Residuals vector of shape (n_observations * 4,). This vector contains the differences 
                    between the predicted and observed pixel coordinates for both left and right images.
    """
    # Unpack camera parameters and 3D points from the parameter vector.
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    point_params = params[n_cameras * 6:].reshape(n_points, 3)
    residuals = []
    
    for cam_idx, point_idx, obs in zip(camera_indices, point_indices, observations):
        # Extract camera pose.
        camera_pose = camera_params[cam_idx]
        angle_axis = camera_pose[:3].reshape(3, 1)
        translation = camera_pose[3:].reshape(3, 1)
        
        # Extract 3D point.
        point = point_params[point_idx].reshape(3, 1)
        
        # Transform the point into the camera coordinate system.
        p_cam = rotate_point(angle_axis, point) + translation
        
        # Convert to homogeneous coordinates.
        p_cam_homogeneous = np.vstack((p_cam, [[1]]))
        
        # Project into left image.
        p_left = P_left @ p_cam_homogeneous
        p_left /= p_left[2]
        predicted_left = p_left[:2].flatten()
        
        # Project into right image.
        p_right = P_right @ p_cam_homogeneous
        p_right /= p_right[2]
        predicted_right = p_right[:2].flatten()
        
        # Each observation is [x_left, y_left, x_right, y_right]
        res = np.hstack((predicted_left - obs[:2], predicted_right - obs[2:]))
        residuals.append(res)
    
    return np.concatenate(residuals)

def run_bundle_adjustment(initial_params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right, max_iter=100):
    """
    Run bundle adjustment using least squares optimization.
    
    Args:
        initial_params (np.ndarray): Initial guess for the parameters.
        n_cameras (int): Number of cameras.
        n_points (int): Number of points.
        camera_indices (np.ndarray): Array mapping observations to camera indices.
        point_indices (np.ndarray): Array mapping observations to point indices.
        observations (np.ndarray): Observed pixel coordinates.
        P_left (np.ndarray): Left camera projection matrix.
        P_right (np.ndarray): Right camera projection matrix.
        
    Returns:
        np.ndarray: Optimized parameters after bundle adjustment.
    """
    result = least_squares(ba_residuals, initial_params,
                           args=(n_cameras, n_points, camera_indices, point_indices,
                                 observations, P_left, P_right))
    return result.x

if __name__ == '__main__':
    # Define a rotation: 45 degrees (pi/4) around the z-axis
    angle_axis = np.array([[np.pi/2], [0], [np.pi/4]])
    
    # Define a point in 3D space
    point = np.array([[1], [0], [0]])
    
    # Rotate the point
    rotated = rotate_point(angle_axis, point)
    print("Original point:", point)
    print("Rotated point:", rotated)
    