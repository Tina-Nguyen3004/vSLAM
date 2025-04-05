from feature_matching import *

def apply_ransac_eight_point(pts1, pts2, K, threshold=1.0, iterations=1000):
    """
    Apply RANSAC to estimate the essential matrix.
    Args:
        pts1 (np.ndarray): Nx2 points from the first image.
        pts2 (np.ndarray): Nx2 points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        threshold (float): Distance threshold for inliers.
        iterations (int): Number of RANSAC iterations.
        
    Returns:
        best_E (np.ndarray): Best estimated essential matrix.
        best_inliers_mask (np.ndarray): Mask of inliers.
    """
    # random sample 8 points
    pts1_norm = normalize_points(pts1, K)
    pts2_norm = normalize_points(pts2, K)
    
    best_inliers_count = 0
    best_E = None
    best_inliners_mask = None
    num_points = pts1_norm.shape[0]
    
    for _ in range(iterations):
        # Randomly sample 8 points
        sample_indices = np.random.choice(num_points, 8, replace=False)
        pts1_sample = pts1_norm[sample_indices]
        pts2_sample = pts2_norm[sample_indices]
        # Compute the essential matrix
        E_candidate = compute_essential_matrix(pts1_sample, pts2_sample)
        
        # Evaluate the candidate by computing the epipolar error
        errors = []
        for i in range(num_points):
            x1 = np.array([pts1_norm[i, 0], pts1_norm[i, 1], 1])
            x2 = np.array([pts2_norm[i, 0], pts2_norm[i, 1], 1])
            error = np.abs(x2.T @ E_candidate @ x1)
            errors.append(error)
        errors = np.array(errors)
        # Count inliers
        inliers = errors < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inliers_count:
            best_inliers_count = inlier_count
            best_E = E_candidate
            best_inliers_mask = inliers
            
    # Return the best essential matrix and inliers
    if best_inliers_mask is not None and inlier_count > 8:
        inlier_pts1 = pts1[best_inliers_mask]
        inlier_pts2 = pts2[best_inliers_mask]
        best_E = compute_essential_matrix(inlier_pts1, inlier_pts2)
        
    return best_E, best_inliers_mask

def normalize_points(points, K):
    """
    Normalize the points using the camera intrinsic matrix.
    Args:
        points (np.ndarray): Nx2 points to be normalized.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        
    Returns:
        normalized_points (np.ndarray): Nx2 normalized points.
    """
    # Invert the camera matrix
    K_inv = np.linalg.inv(K)
    # Normalize the points
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = np.dot(K_inv, homogeneous_points.T)
    normalized_points /= normalized_points[2, :]
    return normalized_points[:2, :].T

def compute_essential_matrix(pts1, pts2):
    """
    Compute the essential matrix using the normalized eight-point algorithm.
    Args: 
        pts1 (np.ndarray): Nx2 normalized points from the first image.
        pts2 (np.ndarray): Nx2 normalized points from the second image.
        
    Returns:
        E (np.ndarray): Essential matrix.
    """
    A = []
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])
    
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    E = U @ np.diag(S) @ Vt
    
    E_cv2 = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0,0), threshold=1.0)[0]
    print("OpenCV Essential Matrix:\n", E_cv2)
    print("Computed Essential Matrix:\n", E)
    return E

if __name__ == "__main__":
    dataset_path = "00"
    frame_index = 0
    
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
    E, inliers_mask = apply_ransac_eight_point(pts1, pts2, K)
    
    
    E_cv, mask_cv = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    E_custom = E/np.linalg.norm(E)
    E_cv = E_cv/np.linalg.norm(E_cv)
    print("OpenCV Essential Matrix:\n", E_cv)
    print("Custom Essential Matrix:\n", E_custom)
    
    # img_matches = draw_matches(img_left_prev, kp_left_prev, img_left_curr, kp_left, matches, 40)
    img_left_curr = draw_inlier_points(img_left_curr, pts2, mask_cv)
    img_left_prev = draw_inlier_points(img_left_prev, pts1, mask_cv)
    
    # Display the images
    # cv2.imshow("Matches", img_matches)
    # cv2.waitKey(0)
    cv2.imshow("Inliers", img_left_curr)
    cv2.imshow("Inliers Previous", img_left_prev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()