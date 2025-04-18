from feature_matching import *

import numpy as np
import numpy.linalg as la

import cv2
import math

def compute_normalized_points(points):
    """
    Normalize the points using Hartley normalization.
    Shifts the points so that their centroid is at the origin and scales them so that
    the average distance from the origin is sqrt(2).
    
    Args:
        points (np.ndarray): Nx2 array of points (pixel coordinates).
        
    Returns:
        normalized (np.ndarray): Nx2 array of normalized points.
        T (np.ndarray): 3x3 transformation matrix used for normalization.
    """
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    avg_dist = np.mean(np.linalg.norm(shifted_points, axis=1))
    scale = np.sqrt(2) / avg_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized = (T @ homogeneous_points.T).T
    return normalized[:, :2], T

def compute_fundamental_matrix(pts1, pts2):
    """
    Compute the fundamental matrix using the normalized eight-point algorithm.
    Args:
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
        
    Returns:
        F (np.ndarray): 3x3 fundamental matrix.
    """
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    return F

def compute_essential_matrix(pts1, pts2, K):
    """
    Compute the essential matrix using improved normalization.
    Args:
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        
    Returns:
        E (np.ndarray): 3x3 essential matrix.
    """
    pts1_norm, T1 = compute_normalized_points(pts1)
    pts2_norm, T2 = compute_normalized_points(pts2)
    F_norm = compute_fundamental_matrix(pts1_norm, pts2_norm)
    F = T2.T @ F_norm @ T1
    E = K.T @ F @ K
    return E

def sampson_distance(F, pt1, pt2):
    """
    Compute the squared Sampson distance between two points given a fundamental matrix.
    
    Args:
        F (np.ndarray): 3x3 fundamental matrix.
        pt1 (np.ndarray): 3x1 point in the first image (homogeneous coordinates).
        pt2 (np.ndarray): 3x1 point in the second image (homogeneous coordinates).
        
    Returns:
        distance (float): The Sampson distance.
    """
    num = (pt2.T @ F @ pt1) ** 2
    Fx1 = F @ pt1
    Ftx2 = F.T @ pt2
    denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    if denom < 1e-6:
        return np.inf
    return float(num / denom)

def apply_ransac_eight_point(pts1, pts2, K, threshold=5.0, desired_confidence=0.90, max_iterations=1000):
    """
    Custom RANSAC implementation to estimate the essential matrix using unnormalized points.
    This version uses vectorized error computation and dynamically adjusts the number of iterations,
    similar in spirit to OpenCV’s approach.
    
    Args:
        pts1 (np.ndarray): Nx2 raw points from the first image.
        pts2 (np.ndarray): Nx2 raw points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        threshold (float): Inlier threshold for the Sampson distance.
        desired_confidence (float): Desired overall confidence (e.g., 0.99).
        max_iterations (int): Maximum number of iterations.
        
    Returns:
        best_E (np.ndarray): The best estimated 3x3 essential matrix.
        best_inlier_mask (np.ndarray): Boolean mask (length N) of inliers.
        actual_iterations (int): The number of iterations run.
    """
    best_E = None
    best_inlier_mask = None
    best_inlier_count = 0
    num_points = pts1.shape[0]
    n = 8  # sample size

    current_iteration = 0
    required_iterations = max_iterations

    # Precompute homogeneous coordinates for all points.
    pts1_h = np.hstack((pts1, np.ones((num_points, 1))))
    pts2_h = np.hstack((pts2, np.ones((num_points, 1))))

    while current_iteration < required_iterations:
        current_iteration += 1
        sample_indices = np.random.choice(num_points, n, replace=False)
        pts1_sample = pts1[sample_indices]
        pts2_sample = pts2[sample_indices]
        E_candidate = compute_essential_matrix(pts1_sample, pts2_sample, K)
        
        # Vectorized computation of Sampson error:
        Fx1 = E_candidate @ pts1_h.T  # Shape: [3, num_points]
        Ftx2 = E_candidate.T @ pts2_h.T  # Shape: [3, num_points]
        numerator = np.sum(pts2_h * (E_candidate @ pts1_h.T).T, axis=1) ** 2
        denominator = Fx1[0, :]**2 + Fx1[1, :]**2 + Ftx2[0, :]**2 + Ftx2[1, :]**2
        denominator[denominator < 1e-6] = 1e-6
        errors = numerator / denominator

        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_E = E_candidate
            best_inlier_mask = inlier_mask

            if best_inlier_count > 0:
                w = best_inlier_count / float(num_points)
                # If nearly all points are inliers, break early.
                if w ** n >= 0:
                    required_iterations = current_iteration
                else:
                    print(math.log(1 - w ** n), w, n)
                    required_iterations = min(max_iterations,
                        int(math.ceil(math.log(1 - desired_confidence) / math.log(1 - w ** n))))
            # Otherwise (best_inlier_count==0), leave required_iterations unchanged.
    
    # Optionally, refine the essential matrix using all inliers.
    if best_inlier_mask is not None and best_inlier_count > n:
        inlier_pts1 = pts1[best_inlier_mask]
        inlier_pts2 = pts2[best_inlier_mask]
        best_E = compute_essential_matrix(inlier_pts1, inlier_pts2, K)
        
    return best_E, best_inlier_mask, current_iteration

def triangulate_for_points(P1, P2, pts1, pts2):
    """
    Triangulate points given two projection matrices and corresponding points.
    
    Args:
        P1 (np.ndarray): 3x4 projection matrix for the first camera.
        P2 (np.ndarray): 3x4 projection matrix for the second camera.
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
    Returns:
        points_3D (np.ndarray): 4xN array of triangulated 3D points.
    """
    num_points = pts1.shape[0]
    points_3D = np.zeros((num_points, 4))

    # Create the A matrix for all points
    for i in range(num_points):
        
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points_3D[i] = X
    return points_3D.T

def recover_pose(E, pts1, pts2, K):
    """
    Recover the relative camera pose from the essential matrix.
    
    Args:
        E (np.ndarray): 3x3 essential matrix.
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        
    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    # Decompose the essential matrix
    U, _, Vt = np.linalg.svd(E)
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Compute the two candidate rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    # Ensure that rotations have a positive determinant
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        
    # The translation is the third column of U
    t = U[:, 2]
    candidates = [
        (R1, t), 
        (R2, t), 
        (R1, -t), 
        (R2, -t)]
    
    # Prepare the projection matrix for the first camera
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    best_inliner_count = -1
    best_R, best_t = None, None
    
    for R_candidate, t_candidate in candidates:
        # Prepare the projection matrix for the second consecutive image
        P2 = K @ np.hstack((R_candidate, t_candidate.reshape(-1, 1)))
        # Triangulate points
        points_3D = triangulate_for_points(P1, P2, pts1, pts2)
        pos_dep1 = np.sum(points_3D[2] > 0)
        
        # For camera 2, we need to check the sign of the Z coordinate
        pts3d_cam2 = R_candidate @ points_3D[:3] + t_candidate.reshape(-1, 1)
        pos_dep2 = np.sum(pts3d_cam2[2] > 0)

        inliner_count = min(pos_dep1, pos_dep2)
        if inliner_count > best_inliner_count:
            best_inliner_count = inliner_count
            best_R, best_t = R_candidate, t_candidate
        
    return best_R, best_t

def main():
    # Synthetic example:
    np.random.seed(42)
    num_points = 50
    pts1 = np.random.uniform(low=0, high=640, size=(num_points, 2))
    
    # Known transformation: small rotation and translation.
    angle = np.deg2rad(5)
    R_true = np.array([[math.cos(angle), -math.sin(angle), 0],
                       [math.sin(angle),  math.cos(angle), 0],
                       [0,                0,               1]])
    t_true = np.array([[10], [5], [1]])
    
    # Define camera intrinsic matrix K.
    K = np.array([[700,   0, 320],
                  [  0, 700, 240],
                  [  0,   0,   1]], dtype=float)
    
    # Generate pts2 by applying the transformation to pts1 (simulate a pure 2D reprojection).
    pts2 = []
    for pt in pts1:
        pt_h = np.array([[pt[0]], [pt[1]], [1]])
        pt2_h = R_true @ pt_h + t_true
        pt2 = (pt2_h / pt2_h[2]).flatten()[:2]
        pts2.append(pt2)
    pts2 = np.array(pts2)
    
    # Add noise to pts2.
    pts2 += np.random.normal(scale=1.0, size=pts2.shape)
    
    # Run custom RANSAC eight-point algorithm.
    E_custom, mask_custom, iters_custom = apply_ransac_eight_point(pts1, pts2, K,
                                                                   threshold=5.0,
                                                                   desired_confidence=0.90,
                                                                   max_iterations=1000)
    print("Custom RANSAC (eight-point) estimated essential matrix:")
    print(E_custom)
    print("Custom inlier mask:")
    print(mask_custom)
    print("Iterations (custom):", iters_custom)
    
    # Use OpenCV's findEssentialMat.
    pts1_cv = pts1.astype(np.float32)
    pts2_cv = pts2.astype(np.float32)
    E_cv, mask_cv = cv2.findEssentialMat(pts1_cv, pts2_cv, K, method=cv2.RANSAC,
                                          prob=0.999, threshold=1.0)
    print("\nOpenCV findEssentialMat estimated essential matrix:")
    print(E_cv)
    print("OpenCV inlier mask:")
    print(mask_cv.ravel())
    
    # Optional: Recover pose from one of the estimates.
    R_custom, t_custom = recover_pose(E_custom, pts1, pts2, K)
    R_cv, t_cv = cv2.recoverPose(E_cv, pts1_cv, pts2_cv, K)[1:3]
    print("\nRecovered pose (custom):")
    print("Rotation:\n", R_custom)
    print("Translation:\n", t_custom)
    print("\nRecovered pose (OpenCV):")
    print("Rotation:\n", R_cv)
    print("Translation:\n", t_cv)
    
    # Compare the two essential matrices (note: defined up to scale).
    E_custom_norm = E_custom / la.norm(E_custom)
    E_cv_norm = E_cv / la.norm(E_cv)
    diff = la.norm(E_custom_norm - E_cv_norm)
    print("\nFrobenius norm difference between normalized matrices:")
    print(diff)
    
if __name__ == "__main__":
    main()
    