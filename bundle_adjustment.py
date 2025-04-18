import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def rotate_points(angle_axes, points):
    """
    Rotate multiple 3D points using Rodrigues' rotation formula (vectorized).
    
    Args:
        angle_axes (np.ndarray): (n, 3) array of axis-angle rotations.
        points (np.ndarray): (n, 3) array of 3D points.
    
    Returns:
        np.ndarray: (n, 3) array of rotated points.
    """
    angle_axes = np.asarray(angle_axes, dtype=float)
    points = np.asarray(points, dtype=float)
    
    # Compute the rotation magnitudes (theta)
    thetas = np.linalg.norm(angle_axes, axis=1, keepdims=True)  # (n, 1)
    
    # Normalize the rotation axes (k)
    k = np.divide(angle_axes, thetas, where=thetas > 1e-8)  # Avoid division by zero
    k = np.nan_to_num(k)  # Handle cases where theta is close to zero
    
    # Compute cos(theta) and sin(theta)
    cos_thetas = np.cos(thetas)  # (n, 1)
    sin_thetas = np.sin(thetas)  # (n, 1)
    
    # Compute cross product (k x points) and dot product (k . points)
    k_cross_p = np.cross(k, points)  # (n, 3)
    k_dot_p = np.sum(k * points, axis=1, keepdims=True)  # (n, 1)
    
    # Apply Rodrigues' rotation formula
    rotated_points = points * cos_thetas + k_cross_p * sin_thetas + k * (k_dot_p * (1 - cos_thetas))
    return rotated_points

def project_point_monocular(point, K):
    """
    Project a 3D point to 2D using the camera intrinsic matrix K.
    """
    point = np.asarray(point, dtype=float).reshape(3, 1)
    projected = K @ point
    projected /= projected[2]
    return projected[:2]

def ba_residuals_monocular(params, n_cameras, n_points, camera_indices, point_indices, observations, K):
    """
    Compute the reprojection residuals for monocular bundle adjustment (fully vectorized).
    """
    # Reshape camera and point parameters
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)  # (n_cameras, 6)
    point_params = params[n_cameras * 6:].reshape(n_points, 3)    # (n_points, 3)

    # Extract angle-axis and translation for all cameras
    angle_axes = camera_params[:, :3]  # (n_cameras, 3)
    translations = camera_params[:, 3:]  # (n_cameras, 3)

    # Extract points corresponding to observations
    points = point_params[point_indices]  # (n_observations, 3)
    angle_axes_obs = angle_axes[camera_indices]  # (n_observations, 3)
    translations_obs = translations[camera_indices]  # (n_observations, 3)

    # Rotate points using the vectorized Rodrigues' formula
    rotated_points = rotate_points(angle_axes_obs, points)  # (n_observations, 3)

    # Transform points into the camera coordinate frame
    p_cam = rotated_points + translations_obs  # (n_observations, 3)

    # Project points to 2D
    projected_points = (K @ p_cam.T).T  # (n_observations, 3)
    projected_points = projected_points[:, :2] / projected_points[:, 2:]  # Normalize to 2D

    # Compute residuals
    residuals = projected_points - observations  # (n_observations, 2)

    return residuals.flatten()

def finite_difference_jacobian(func, x, eps, *args):
    """
    Approximate the Jacobian of a function at x using finite differences (vectorized).
    
    Args:
        func: function that takes x (1D array) and returns a 1D residual array.
        x: point at which to evaluate the Jacobian.
        eps: perturbation for finite differences.
        args: additional arguments passed to func.
        
    Returns:
        Jacobian matrix of shape (m, n) where m = len(func(x)) and n = len(x).
    """
    x = np.asarray(x, dtype=float)
    f0 = func(x, *args)  # Evaluate the function at the original point
    n = x.size
    m = f0.size
    
    # Create a perturbation matrix
    perturbations = np.eye(n) * eps  # Shape: (n, n)
    
    # Add perturbations to x
    x_perturbed = x + perturbations.T  # Shape: (n, n)
    
    # Evaluate the function at all perturbed points
    f_perturbed = np.array([func(x_perturbed[:, i], *args) for i in range(n)]).T  # Shape: (m, n)
    
    # Compute the Jacobian using finite differences
    J = (f_perturbed - f0[:, np.newaxis]) / eps  # Shape: (m, n)
    
    return J

def my_least_squares(func, x0, args=(), max_nfev=100, tol=1e-6, eps=1e-6):
    """
    A simple Gauss-Newton least-squares optimizer using finite differences for the Jacobian.
    """
    x = np.asarray(x0, dtype=float)
    lambda_reg = 1e-6  # Regularization parameter
    for i in range(max_nfev):
        # Evaluate residuals and Jacobian
        f = func(x, *args)
        J = finite_difference_jacobian(func, x, eps, *args)
        
        # Solve normal equations with regularization
        A = J.T @ J + lambda_reg * np.eye(J.shape[1])  # Add regularization
        g = J.T @ f
        try:
            dx = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Stopping optimization.")
            break
        
        x = x + dx
        
        # Debug: Print update norm and residual norm if desired
        # print(f"Iter {i}: ||dx|| = {np.linalg.norm(dx):.6f}, ||f|| = {np.linalg.norm(f):.6f}")
        
        if np.linalg.norm(dx) < tol:
            break
    return {'x': x}

def run_bundle_adjustment_monocular(initial_params, n_cameras, n_points, camera_indices, point_indices, observations, K, max_iter=100):
    """
    Run monocular bundle adjustment using the custom least-squares optimizer.
    
    Args:
        initial_params (np.ndarray): Initial guess, shape (n_cameras*6 + n_points*3,).
        n_cameras (int): Number of cameras (keyframes).
        n_points (int): Number of 3D points.
        camera_indices (np.ndarray): Array mapping observations to camera indices.
        point_indices (np.ndarray): Array mapping observations to 3D point indices.
        observations (np.ndarray): Observed pixel coordinates, shape (n_observations, 2).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        max_iter (int): Maximum iterations for our optimizer.
        
    Returns:
        Optimized parameter vector.
    """
    # Our custom least squares optimizer is used instead of SciPy's.
    result = my_least_squares(ba_residuals_monocular, initial_params,
                              args=(n_cameras, n_points, camera_indices, point_indices, observations, K),
                              max_nfev=max_iter)
    return result['x']

def integrate_monocular_ba(keyframe_db, landmarks_db, observations_db, K):
    """
    Integrate the monocular bundle adjustment into the VSLAM pipeline.
    
    Args:
        keyframe_db (list): List of keyframe dictionaries with 'id' and 'pose' (4x4 matrix).
        landmarks_db (list): List of initial 3D landmark positions (each (3,) np.array).
        observations_db (list): List of observation dictionaries. Each observation should have:
                                - 'camera_idx': index of keyframe
                                - 'point_idx': index of 3D point in landmarks_db
                                - 'observation': np.array([x, y])
        K (np.ndarray): Camera intrinsic matrix (3x3)
    
    Returns:
        Optimized parameter vector containing refined camera poses and 3D points.
    """
    n_cameras = len(keyframe_db)
    n_points = len(landmarks_db)
    
    # Assemble camera parameters from keyframe poses (convert 4x4 to 6 parameters: angle-axis + t)
    camera_params = []
    for kf in keyframe_db:
        pose = kf['pose']  # 4x4 homogeneous matrix
        R_mat = pose[:3, :3]
        t = pose[:3, 3]
        r = R.from_matrix(R_mat)
        angle_axis = r.as_rotvec()  # 3-element vector
        cam_param = np.hstack((angle_axis, t))
        camera_params.append(cam_param)
    camera_params = np.array(camera_params).reshape(-1)
    
    # Assemble landmarks.
    point_params = np.array(landmarks_db).reshape(-1)
    
    # Combine camera and point parameters.
    initial_params = np.hstack((camera_params, point_params))
    
    # Prepare indices and observation array.
    camera_indices = []
    point_indices = []
    observations = []  # Each is a 2D point [x, y]
    for obs in observations_db:
        camera_indices.append(obs['camera_idx'])
        point_indices.append(obs['point_idx'])
        observations.append(obs['observation'])
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    observations = np.array(observations)
    
    # Run our custom bundle adjustment solver.
    optimized_params = run_bundle_adjustment_monocular(initial_params,
                                                       n_cameras,
                                                       n_points,
                                                       camera_indices,
                                                       point_indices,
                                                       observations,
                                                       K,
                                                       max_iter=100)
    return optimized_params

def update_keyframes_from_ba(keyframe_db, optimized_params):
    """
    Update the keyframe poses from the optimized bundle adjustment parameters.
    
    Args:
        keyframe_db (list): List of keyframe dictionaries.
        optimized_params (np.ndarray): Optimized parameter vector.
    """
    n_cameras = len(keyframe_db)
    camera_params_optimized = optimized_params[:n_cameras * 6].reshape(n_cameras, 6)
    for idx, kf in enumerate(keyframe_db):
        angle_axis = camera_params_optimized[idx, :3]
        translation = camera_params_optimized[idx, 3:]
        R_mat = R.from_rotvec(angle_axis).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = R_mat
        pose[:3, 3] = translation
        kf['pose'] = pose
        print(f"Keyframe {kf['id']} updated pose:\n{kf['pose']}")

# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Here you would set up your keyframe_db, landmarks_db, and observations_db.
    # For demonstration purposes, we'll define dummy variables.
    
    # Assume a simple intrinsic matrix.
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])
    
    # Dummy keyframe database (2 keyframes with identity and a small translation).
    keyframe_db = [
        {'id': 0, 'pose': np.eye(4)},
        {'id': 1, 'pose': np.array([[0.9998, 0.0176, 0.0053, 0.1],
                                     [-0.0175, 0.9998, 0.0123, 0.0],
                                     [-0.0055, -0.0122, 0.9999, 0.0],
                                     [0, 0, 0, 1]])}
    ]
    
    # Dummy landmarks (a few 3D points).
    landmarks_db = [
        np.array([1, 2, 10]),
        np.array([2, 1, 12]),
        np.array([0, 0, 8])
    ]
    
    # Dummy observations: for each observation we indicate the keyframe (camera_idx),
    # landmark (point_idx), and a 2D observation (in pixels).
    observations_db = [
        {'camera_idx': 0, 'point_idx': 0, 'observation': np.array([620, 240])},
        {'camera_idx': 0, 'point_idx': 1, 'observation': np.array([600, 230])},
        {'camera_idx': 1, 'point_idx': 0, 'observation': np.array([622, 241])},
        {'camera_idx': 1, 'point_idx': 2, 'observation': np.array([610, 250])}
    ]
    
    # Run our monocular bundle adjustment.
    optimized_params = integrate_monocular_ba(keyframe_db, landmarks_db, observations_db, K)
    
    # Update keyframe poses based on the BA results.
    update_keyframes_from_ba(keyframe_db, optimized_params)
    
    # Optionally, you can update your landmarks_db from optimized_params as well.
    n_cameras = len(keyframe_db)
    optimized_landmarks = optimized_params[n_cameras * 6:].reshape(len(landmarks_db), 3)
    print("Optimized landmarks:")
    print(optimized_landmarks)