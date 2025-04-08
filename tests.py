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

if __name__ == '__main__':
    test_ba_residuals()