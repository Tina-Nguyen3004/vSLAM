from bundle_adjustment import *

def test_ba_residuals():
    """
    Test the ba_residuals function with a simple case:
      - One camera (n_cameras = 1)
      - One 3D point (n_points = 1)
      - The camera has zero rotation and zero translation.
      - The 3D point is located at (0, 0, 5).
      - Left projection matrix: P_left = K [I|0] with K = I (identity intrinsic).
      - Right projection matrix: P_right = K [I | -baseline] with baseline = 1.
      
    Expected:
      - The left projection of the point should be [0, 0].
      - The right projection should be computed as follows:
          * The 3D point in the right camera is [0 - baseline, 0, 5] = [-1, 0, 5].
          * Its projection is [-1/5, 0] = [-0.2, 0].
      - Observations are given as [0, 0, -0.2, 0].
      - Residuals should be close to zero.
    """
    # Setup parameters.
    n_cameras = 1
    n_points = 1
    
    # Camera: no rotation, no translation.
    camera_pose = np.array([0, 0, 0, 0, 0, 0])  # shape (6,)
    
    # 3D point: at (0, 0, 5)
    point = np.array([0, 0, 5])  # shape (3,)
    
    # Combine parameters.
    params = np.hstack((camera_pose, point))
    
    # Observation indices.
    camera_indices = np.array([0])
    point_indices = np.array([0])
    
    # Define intrinsic matrix K as identity (for simplicity).
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    # Define projection matrices.
    P_left = np.hstack((K, np.zeros((3, 1))))   # P_left = [I|0]
    baseline = 1.0
    P_right = np.hstack((K, np.array([[-baseline], [0], [0]])))  # P_right = [I| -baseline]
    
    # Expected projections:
    # Left: [0/5, 0/5] = [0, 0]
    # Right: [(-1)/5, 0/5] = [-0.2, 0]
    observations = np.array([[0, 0, -0.2, 0]])  # shape (1, 4)
    
    # Compute residuals.
    residuals = ba_residuals(params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right)
    
    print("Residuals:", residuals)
    
    # The residuals should be very close to zero.
    assert np.allclose(residuals, np.zeros_like(residuals), atol=1e-8), "Test failed: residuals are not zero"
    print("Test passed.")

def test_ba_residuals_hybrid():
    """
    Test the ba_residuals function with a hybrid scenario:
      - Two cameras (n_cameras = 2) and three 3D points (n_points = 3).
      - Camera 0: zero rotation, zero translation.
      - Camera 1: rotated about z by 10 degrees and translated by (0.2, 0, 0).
      - 3D points:
          * p0 = [1, 1, 8]
          * p1 = [-1, 2, 10]
          * p2 = [0, -1, 7]
      - Left projection matrix: P_left = K[I | 0] with K = I.
      - Right projection matrix: P_right = K[I | -baseline] with baseline = 1.
      
    For each camera-point pair, we generate an observation using the forward model.
    When running ba_residuals with these exact parameters and observations,
    the residuals should be close to zero.
    """
    n_cameras = 2
    n_points = 3

    # Define camera poses.
    # Camera 0: no rotation, no translation.
    camera0 = np.array([0, 0, 0, 0, 0, 0])
    # Camera 1: rotation about z-axis (10 degrees), translation (0.2, 0, 0)
    angle_axis_cam1 = np.array([0, 0, np.deg2rad(10)])  # 10 degrees around z
    translation_cam1 = np.array([0.2, 0, 0])
    camera1 = np.hstack((angle_axis_cam1, translation_cam1))
    camera_params = np.vstack((camera0, camera1))  # shape (2, 6)

    # Define 3D points.
    p0 = np.array([1, 1, 8])
    p1 = np.array([-1, 2, 10])
    p2 = np.array([0, -1, 7])
    point_params = np.vstack((p0, p1, p2))  # shape (3, 3)

    # Pack initial parameters.
    initial_params = np.hstack((camera_params.ravel(), point_params.ravel()))
    
    # Build observation mapping.
    # Assume each camera observes each point.
    # Total observations = 2 cameras * 3 points = 6.
    camera_indices = np.repeat(np.arange(n_cameras), n_points)  # [0,0,0, 1,1,1]
    point_indices = np.tile(np.arange(n_points), n_cameras)       # [0,1,2, 0,1,2]
    
    # Set intrinsic matrix as identity.
    K = np.eye(3)
    # Define projection matrices.
    P_left = np.hstack((K, np.zeros((3, 1))))  # [I | 0]
    baseline = 1.0
    P_right = np.hstack((K, np.array([[-baseline], [0], [0]])))  # [I | -baseline]
    
    # Now simulate observations for each camera-point pair.
    observations = []
    for cam_idx, pt_idx in zip(camera_indices, point_indices):
        # Extract the corresponding camera pose.
        cam_pose = camera_params[cam_idx]
        angle_axis = cam_pose[:3].reshape(3, 1)
        translation = cam_pose[3:].reshape(3, 1)
        # Extract the 3D point.
        pt = point_params[pt_idx].reshape(3, 1)
        p_cam = rotate_point(angle_axis, pt) + translation
        p_cam_hom = np.vstack((p_cam, [[1]]))
        # Project using left camera.
        p_left = P_left @ p_cam_hom
        p_left /= p_left[2]
        pred_left = p_left[:2].flatten()
        # Project using right camera.
        p_right = P_right @ p_cam_hom
        p_right /= p_right[2]
        pred_right = p_right[:2].flatten()
        # Observation is [x_left, y_left, x_right, y_right].
        observations.append(np.hstack((pred_left, pred_right)))
    observations = np.array(observations)  # shape (6, 4)
    
    # Now call ba_residuals with our parameters and simulated observations.
    residuals = ba_residuals(initial_params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right)
    
    print("Hybrid Test Residuals:", residuals)
    # Check that residuals are (close to) zero.
    assert np.allclose(residuals, np.zeros_like(residuals), atol=1e-8), "Test failed: Residuals are not zero."
    print("Hybrid test passed: Residuals are near zero.")

def test_ba_residuals_hybrid():
    """
    Test the ba_residuals function with a hybrid scenario:
      - Two cameras (n_cameras = 2) and three 3D points (n_points = 3).
      - Camera 0: zero rotation, zero translation.
      - Camera 1: rotated about z by 10 degrees and translated by (0.2, 0, 0).
      - 3D points:
          * p0 = [1, 1, 8]
          * p1 = [-1, 2, 10]
          * p2 = [0, -1, 7]
      - Left projection matrix: P_left = K[I | 0] with K = I.
      - Right projection matrix: P_right = K[I | -baseline] with baseline = 1.
      
    For each camera-point pair, we generate an observation using the forward model.
    When running ba_residuals with these exact parameters and observations,
    the residuals should be close to zero.
    """
    n_cameras = 2
    n_points = 3

    # Define camera poses.
    # Camera 0: no rotation, no translation.
    camera0 = np.array([0, 0, 0, 0, 0, 0])
    # Camera 1: rotation about z-axis (10 degrees), translation (0.2, 0, 0)
    angle_axis_cam1 = np.array([0, 0, np.deg2rad(10)])  # 10 degrees around z
    translation_cam1 = np.array([0.2, 0, 0])
    camera1 = np.hstack((angle_axis_cam1, translation_cam1))
    camera_params = np.vstack((camera0, camera1))  # shape (2, 6)

    # Define 3D points.
    p0 = np.array([1, 1, 8])
    p1 = np.array([-1, 2, 10])
    p2 = np.array([0, -1, 7])
    point_params = np.vstack((p0, p1, p2))  # shape (3, 3)

    # Pack initial parameters.
    initial_params = np.hstack((camera_params.ravel(), point_params.ravel()))
    
    # Build observation mapping.
    # Assume each camera observes each point.
    # Total observations = 2 cameras * 3 points = 6.
    camera_indices = np.repeat(np.arange(n_cameras), n_points)  # [0,0,0, 1,1,1]
    point_indices = np.tile(np.arange(n_points), n_cameras)       # [0,1,2, 0,1,2]
    
    # Set intrinsic matrix as identity.
    K = np.eye(3)
    # Define projection matrices.
    P_left = np.hstack((K, np.zeros((3, 1))))  # [I | 0]
    baseline = 1.0
    P_right = np.hstack((K, np.array([[-baseline], [0], [0]])))  # [I | -baseline]
    
    # Now simulate observations for each camera-point pair.
    observations = []
    for cam_idx, pt_idx in zip(camera_indices, point_indices):
        # Extract the corresponding camera pose.
        cam_pose = camera_params[cam_idx]
        angle_axis = cam_pose[:3].reshape(3, 1)
        translation = cam_pose[3:].reshape(3, 1)
        # Extract the 3D point.
        pt = point_params[pt_idx].reshape(3, 1)
        p_cam = rotate_point(angle_axis, pt) + translation
        p_cam_hom = np.vstack((p_cam, [[1]]))
        # Project using left camera.
        p_left = P_left @ p_cam_hom
        p_left /= p_left[2]
        pred_left = p_left[:2].flatten()
        # Project using right camera.
        p_right = P_right @ p_cam_hom
        p_right /= p_right[2]
        pred_right = p_right[:2].flatten()
        # Observation is [x_left, y_left, x_right, y_right].
        observations.append(np.hstack((pred_left, pred_right)))
    observations = np.array(observations)  # shape (6, 4)
    
    # Now call ba_residuals with our parameters and simulated observations.
    residuals = ba_residuals(initial_params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right)
    
    print("Hybrid Test Residuals:", residuals)
    # Check that residuals are (close to) zero.
    assert np.allclose(residuals, np.zeros_like(residuals), atol=1e-8), "Test failed: Residuals are not zero."
    print("Hybrid test passed: Residuals are near zero.")
    
def test_run_bundle_adjustment():
    """
    Test run_bundle_adjustment using a simple scenario:
      - One camera (n_cameras=1) and one 3D point (n_points=1).
      - The camera has zero rotation and zero translation.
      - The 3D point is at (0, 0, 5) in the world coordinate system.
      - The left projection matrix is defined as P_left = K [I | 0] with an identity intrinsic (K=I).
      - The right projection matrix is defined as P_right = K [I | -baseline] with baseline=1.
      
    The expected projections are:
      - Left image: [0, 0] (since 0/5=0).
      - Right image: The point shifts to [-1, 0, 5] so that the projection is [-1/5, 0]=[-0.2, 0].
      
    We add small noise to the initial parameters and then run bundle adjustment.
    After optimization, the residuals should be near zero and the optimized parameters should be close to ground truth.
    """
    n_cameras = 1
    n_points = 1

    # True camera parameters: no rotation (angle_axis=0) and no translation.
    true_camera_pose = np.array([0, 0, 0, 0, 0, 0])
    # True 3D point.
    true_point = np.array([0, 0, 5])
    
    # Ground truth parameter vector.
    true_params = np.hstack((true_camera_pose, true_point))
    
    # Create a noisy initial guess.
    noise_level = 1e-3
    initial_params = true_params + noise_level * np.random.randn(*true_params.shape)
    
    # Observation mappings (only one observation).
    camera_indices = np.array([0])
    point_indices = np.array([0])
    
    # Define intrinsic matrix as identity.
    K = np.eye(3)
    # Define projection matrices.
    P_left = np.hstack((K, np.zeros((3, 1))))   # P_left = [I | 0]
    baseline = 1.0
    P_right = np.hstack((K, np.array([[-baseline], [0], [0]])))  # P_right = [I | -baseline]
    
    # Simulate the expected observations using the true parameters.
    # For the left camera, the projected coordinates are [0/5, 0/5] = [0, 0].
    # For the right camera, the point becomes [0-baseline, 0, 5] = [-1, 0, 5],
    # and the projection is [-1/5, 0] = [-0.2, 0].
    observations = np.array([[0, 0, -0.2, 0]])
    
    # Run the bundle adjustment to optimize the parameters.
    optimized_params = run_bundle_adjustment(initial_params, n_cameras, n_points,
                                             camera_indices, point_indices, observations,
                                             P_left, P_right)
    
    print("Optimized Parameters:", optimized_params)
    
    # Compute the residuals for the optimized parameters.
    final_residuals = ba_residuals(optimized_params, n_cameras, n_points, camera_indices, point_indices, observations, P_left, P_right)
    print("Final Residuals:", optimized_params)
    
    # Check that the residuals are nearly zero.
    assert np.allclose(final_residuals, np.zeros_like(final_residuals), atol=1e-6), "Test failed: Residuals are not near zero!"
    print("Test passed: Optimized parameters yield near-zero residuals.")

if __name__ == '__main__':
    test_ba_residuals_hybrid()
    test_ba_residuals()
    test_run_bundle_adjustment()