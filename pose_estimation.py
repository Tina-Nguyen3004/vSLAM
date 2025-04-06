from feature_matching import *

import numpy as np
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
    # Compute the numberator
    num = (pt2.T @ F @ pt1) ** 2
    # Compute the denominator
    Fx1 = F @ pt1
    Fx2 = F.T @ pt2
    demon = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    # Avoid division by zero
    if demon == 0:
        return np.inf
    return num / demon


def apply_ransac_eight_point(pts1, pts2, K, threshold=5.0, desired_confidence=0.80, max_iterations=1000):
    """
    Custom RANSAC implementation to estimate the essential matrix.
    This function dynamically adjusts the number of iterations based on the desired confidence.
    
    Args:
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        threshold (float): Distance threshold for inliers.
        desired_confidence (float): Desired confidence (e.g., 0.99).
        max_iterations (int): Maximum number of iterations to run.
        
    Returns:
        best_E (np.ndarray): Best estimated essential matrix.
        best_inlier_mask (np.ndarray): Boolean mask of inliers.
        actual_iterations (int): The number of iterations that were actually run.
    """
    best_E = None
    best_inlier_mask = None
    best_inlier_count = -1
    num_points = pts1.shape[0]
    n = 8  # number of points per sample

    current_iteration = 0
    required_iterations = max_iterations  # start with maximum iterations

    while current_iteration < required_iterations:
        current_iteration += 1
        sample_indices = np.random.choice(num_points, n, replace=False)
        pts1_sample = pts1[sample_indices]
        pts2_sample = pts2[sample_indices]
        E_candidate = compute_essential_matrix(pts1_sample, pts2_sample, K)
        
        errors = []
        for i in range(num_points):
            x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
            x2 = np.array([pts2[i, 0], pts2[i, 1], 1])
            error = sampson_distance(E_candidate, x1, x2)
            errors.append(error)
        errors = np.array(errors)
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_E = E_candidate
            best_inlier_mask = inlier_mask
            
            # Update the required iterations based on current inlier ratio
            w = best_inlier_count / num_points
            # Avoid division by zero or log(0)
            if w > 0:
                required_iterations = min(
                    max_iterations, 
                    int(math.ceil(math.log(1 - desired_confidence) / math.log(1 - w**n)))
                )
                
    # Refine the essential matrix using all inliers if possible
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

def recoverPose(E, pts1, pts2, K):
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
        print("pts3d_cam1", pos_dep1)
        # For camera 2, we need to check the sign of the Z coordinate
        pts3d_cam2 = R_candidate @ points_3D[:3] + t_candidate.reshape(-1, 1)
        pos_dep2 = np.sum(pts3d_cam2[2] > 0)
        print("pts3d_cam2", pos_dep2)

        inliner_count = min(pos_dep1, pos_dep2)
        if inliner_count > best_inliner_count:
            best_inliner_count = inliner_count
            best_R, best_t = R_candidate, t_candidate
        
    return best_R, best_t

if __name__ == "__main__":
    dataset_path = "00"
    frame_index = 12
    
    # Initialize the first 
    img_left_prev, _, _ = process_stereo(dataset_path, frame_index)
    img_left_curr, _, _ = process_stereo(dataset_path, frame_index + 1)
    
    kp_left_prev, dp_left_prev = featureExtractor(img_left_prev)
    kp_left, dp_left = featureExtractor(img_left_curr)
    
    points1 = np.array([kp.pt for kp in kp_left_prev])
    points2 = np.array([kp.pt for kp in kp_left])
    intrinsic_path = os.path.join(dataset_path, "calib.txt")
    K, _ = extract_intrinsic_parameters(intrinsic_path, "P0")
    
    matches = match_features(dp_left_prev, dp_left)
    pts1 = np.array([kp_left_prev[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp_left[m.trainIdx].pt for m in matches])
    E, best_inlier_mask, current_iteration = apply_ransac_eight_point(pts1, pts2, K)

    R, t = recoverPose(E, pts1, pts2, K)
    print("R", R)
    print("t", t)
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)
    _, R_cv, t_cv, mask = cv2.recoverPose(E, pts1, pts2, K)
    print("OpenCV R", R_cv)
    print("OpenCV t", t_cv)
    # E_cv, mask_cv = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    # print("OpenCV Essential Matrix:\n", E_cv, np.sum(mask_cv))
    
    # # img_matches = draw_matches(img_left_prev, kp_left_prev, img_left_curr, kp_left, matches, 40)
    # img_left_curr = draw_inlier_points(img_left_curr, pts2, mask_cv)
    # img_left_prev = draw_inlier_points(img_left_prev, pts1, mask_cv)
    
    # # Display the images
    # # cv2.imshow("Matches", img_matches)
    # # cv2.waitKey(0)
    # cv2.imshow("Inliers", img_left_curr)
    # cv2.imshow("Inliers Previous", img_left_prev)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()