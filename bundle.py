import numpy as np
import cv2  # For use with cv2.Rodrigues (if needed)
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def reprojection_residuals(params, n_free, n_points, camera_indices, point_indices, observations, K, fixed_cam_param):
    """
    Compute reprojection residuals for all observations in a vectorized manner,
    using a fixed first camera.
    
    Args:
        params (np.ndarray): 1D optimization vector containing free camera parameters
                             (for cameras 1..n-1) and all 3D points.
        n_free (int): Number of free cameras = total cameras - 1.
        n_points (int): Number of 3D points.
        camera_indices (np.ndarray): Array of shape (n_obs,) mapping each observation to a camera index.
                                     Observations with camera index 0 use the fixed camera parameters.
        point_indices (np.ndarray): Array of shape (n_obs,) mapping each observation to a 3D point index.
        observations (np.ndarray): Array of observed 2D points of shape (n_obs, 2).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        fixed_cam_param (np.ndarray): Fixed camera parameters (6,) for camera 0.
        
    Returns:
        np.ndarray: Residual vector (flattened differences between projected and observed 2D points).
    """
    # Unpack free camera parameters (for cameras 1,...,n)
    free_cam_params = params[:n_free * 6].reshape((n_free, 6))
    points_3d = params[n_free * 6:].reshape((n_points, 3))
    
    # Build a complete list of camera parameters: camera0 is fixed, and cameras 1..n come from free parameters.
    all_cam_params = np.vstack([fixed_cam_param, free_cam_params])  # shape (n_free+1, 6)
    
    # Get per-observation camera parameters with vectorized lookup.
    obs_rvecs = all_cam_params[camera_indices, :3]  # shape (n_obs, 3)
    obs_tvecs = all_cam_params[camera_indices, 3:6]   # shape (n_obs, 3)
    obs_points = points_3d[point_indices]             # shape (n_obs, 3)
    
    # Convert rotation vectors to rotation matrices in a vectorized way.
    rot_matrices = R.from_rotvec(obs_rvecs).as_matrix()  # shape (n_obs, 3, 3)
    
    # Transform 3D points into camera coordinates (vectorized).
    transformed_points = np.einsum('ijk,ik->ij', rot_matrices, obs_points) + obs_tvecs  # shape (n_obs, 3)
    
    # Avoid division-by-zero for points behind or too close to the camera.
    x = transformed_points[:, 0]
    y = transformed_points[:, 1]
    z = transformed_points[:, 2]
    z = np.where(z == 0, 1e-8, z)
    
    # Extract camera intrinsics.
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Compute predicted pixel coordinates.
    u_pred = fx * (x / z) + cx
    v_pred = fy * (y / z) + cy
    predictions = np.column_stack((u_pred, v_pred))
    
    # Compute residuals between predictions and observed feature positions.
    residuals = (predictions - observations).ravel()
    return residuals

def local_bundle_adjustment(recent_keyframes, observations_data, K):
    """
    Perform vectorized local bundle adjustment over a sliding window of keyframes,
    with the first keyframe fixed to remove gauge ambiguity.
    
    Args:
        recent_keyframes (list): List of keyframe dictionaries, each containing:
            - 'id': unique keyframe identifier.
            - 'pose': 4x4 transformation matrix.
        observations_data (dict): Dictionary containing:
            - 'camera_indices': np.ndarray mapping each observation to a camera index.
            - 'point_indices': np.ndarray mapping each observation to a 3D point index.
            - 'observations': np.ndarray of observed 2D points (shape: [n_obs, 2]).
            - 'points_3d': np.ndarray of initial 3D points (each row corresponds to a point).
        K (np.ndarray): Camera intrinsic matrix.
    
    Returns:
        dict: Mapping from keyframe id to optimized 4x4 pose matrices.
              (The first keyframe's pose remains unchanged.)
    """
    n_total = len(recent_keyframes)  # Total cameras/keyframes.
    if n_total < 1:
        return {}
    
    # Fix the first camera (keyframe) to remove the gauge freedom.
    fixed_cam = recent_keyframes[0]
    fixed_R = fixed_cam['pose'][:3, :3]
    fixed_t = fixed_cam['pose'][:3, 3]
    fixed_cam_param = np.hstack((R.from_matrix(fixed_R).as_rotvec(), fixed_t))  # shape (6,)
    
    # Free cameras: for keyframes 1..(n_total - 1)
    free_cam_params = []
    for kf in recent_keyframes[1:]:
        R_mat = kf['pose'][:3, :3]
        tvec = kf['pose'][:3, 3]
        rvec = R.from_matrix(R_mat).as_rotvec()
        free_cam_params.append(np.hstack((rvec, tvec)))
    free_cam_params = np.array(free_cam_params)  # shape (n_total - 1, 6)
    
    points_3d = observations_data['points_3d']  # shape (n_points, 3)
    n_points = points_3d.shape[0]
    n_free = n_total - 1  # number of free cameras
    
    # Build the optimization vector: only free cameras and all 3D points.
    x0 = np.hstack((free_cam_params.flatten(), points_3d.flatten()))
    
    # Unpack observation data.
    camera_indices = observations_data['camera_indices']
    point_indices = observations_data['point_indices']
    observations = observations_data['observations']
    
    # Run the least-squares optimizer.
    result = least_squares(
        reprojection_residuals,
        x0,
        loss = 'huber',
        args=(n_free, n_points, camera_indices, point_indices, observations, K, fixed_cam_param),
        verbose=2
    )
    
    # Extract optimized free camera parameters.
    optimized_free = result.x[:n_free * 6].reshape((n_free, 6))
    
    # Reconstruct the full (optimized) camera poses.
    optimized_poses = {}
    # The first keyframe stays fixed.
    optimized_poses[recent_keyframes[0]['id']] = fixed_cam['pose']
    
    # For the free cameras, convert the parameters back to 4x4 poses.
    for i, kf in enumerate(recent_keyframes[1:]):
        rvec_opt = optimized_free[i, :3]
        tvec_opt = optimized_free[i, 3:6]
        R_opt = R.from_rotvec(rvec_opt).as_matrix()
        T_opt = np.eye(4)
        T_opt[:3, :3] = R_opt
        T_opt[:3, 3] = tvec_opt
        optimized_poses[kf['id']] = T_opt
        
    return optimized_poses

# ===== Example usage (for testing) =====
if __name__ == "__main__":
    # Create dummy keyframes (for illustration).
    recent_keyframes = [
        {'id': 0, 'pose': np.eye(4)},
        {'id': 1, 'pose': np.array([[0.9998, -0.0175, 0.005, 0.1],
                                     [0.0175, 0.9998, -0.003, 0.0],
                                     [-0.005, 0.003, 1.0, 0.0],
                                     [0, 0, 0, 1]])},
        {'id': 2, 'pose': np.array([[0.9997, -0.021, 0.008, 0.2],
                                     [0.021, 0.9997, -0.004, 0.1],
                                     [-0.008, 0.004, 1.0, 0.0],
                                     [0, 0, 0, 1]])}
    ]
    
    # Create dummy observations data.
    observations_data = {
        # Assume 6 observations distributed across 3 keyframes:
        'camera_indices': np.array([0, 0, 1, 1, 2, 2]),  # indices refer to keyframes in recent_keyframes
        'point_indices':  np.array([0, 1, 0, 1, 0, 1]),
        'observations': np.array([
            [320, 240],
            [330, 245],
            [322, 238],
            [332, 244],
            [324, 242],
            [334, 240]
        ], dtype=np.float32),
        # Dummy initial 3D points for two landmarks.
        'points_3d': np.array([
            [1.0, 2.0, 10.0],
            [1.5, 1.8, 9.5]
        ], dtype=np.float32)
    }
    
    # Dummy camera intrinsic matrix.
    K = np.array([
        [718.856,   0,      607.1928],
        [   0,    718.856,  185.2157],
        [   0,       0,         1   ]
    ], dtype=np.float32)
    
    optimized = local_bundle_adjustment(recent_keyframes, observations_data, K)
    for kf_id, pose in optimized.items():
        print(f"Keyframe {kf_id} optimized pose:\n{pose}\n")
