#!/usr/bin/env python3
import cv2
import numpy as np
import math
import numpy.linalg as la

########################################
# Fundamental/Essential Matrix Functions
########################################

def compute_fundamental_matrix(pts1, pts2):
    """
    Compute the fundamental matrix using the eight-point algorithm 
    without normalizing the points.
    
    Args:
        pts1 (np.ndarray): Nx2 raw points from the first image.
        pts2 (np.ndarray): Nx2 raw points from the second image.
        
    Returns:
        F (np.ndarray): 3x3 fundamental matrix with rank-2 enforced.
    """
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([x1*x2, x1*y2, x1,
                  y1*x2, y1*y2, y1,
                  x2,    y2,    1])
    A = np.array(A)
    _, _, Vt = la.svd(A)
    F = Vt[-1].reshape(3, 3)
    # Enforce rank-2 constraint on F.
    U, S, Vt = la.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    return F

def compute_essential_matrix(pts1, pts2, K):
    """
    Compute the essential matrix using raw (unnormalized) point correspondences
    via the eight-point algorithm.
    
    Args:
        pts1 (np.ndarray): Nx2 raw points from the first image.
        pts2 (np.ndarray): Nx2 raw points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        
    Returns:
        E (np.ndarray): 3x3 essential matrix.
    """
    F = compute_fundamental_matrix(pts1, pts2)
    E = K.T @ F @ K
    return E

def sampson_distance(F, pt1, pt2):
    """
    Compute the squared Sampson distance between two points given a fundamental matrix.
    
    Args:
        F (np.ndarray): 3x3 fundamental matrix.
        pt1 (np.ndarray): 3x1 homogeneous point in the first image.
        pt2 (np.ndarray): 3x1 homogeneous point in the second image.
        
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

#############################################
# Optimized Minimal Five-Point Algorithm Prototype
#############################################
def five_point_essential_optimized(pts1, pts2, K, grid_res=10):
    """
    A simplified version of the five-point algorithm that, given exactly 5 correspondences,
    computes the nullspace of the 5x9 measurement matrix and then uses a vectorized grid search
    to find coefficients (a, b, c) such that candidate = a*E0 + b*E1 + c*E2 + E3 minimizes the 
    total epipolar error.
    
    Args:
        pts1 (np.ndarray): 5x2 raw points from the first image.
        pts2 (np.ndarray): 5x2 raw points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        grid_res (int): Number of grid steps in each dimension.
        
    Returns:
        E_est (np.ndarray): 3x3 essential matrix estimate.
    """
    if pts1.shape[0] != 5 or pts2.shape[0] != 5:
        raise ValueError("five_point_essential_optimized requires exactly 5 correspondences.")
    
    # Build the 5x9 design matrix A from the epipolar constraint x2^T*E*x1 = 0.
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    A = np.array(A)
    
    # SVD of A; the nullspace is 4-dimensional.
    U, S, Vt = la.svd(A)
    nullspace = Vt[-4:, :].T  # Shape: (9, 4)
    
    # Each column of nullspace reshaped is a 3x3 basis matrix.
    E_basis = [nullspace[:, i].reshape(3, 3) for i in range(4)]
    
    # Prepare the grid search:
    grid_vals = np.linspace(-1, 1, grid_res)
    # Generate all combinations of (a, b, c) on the grid: shape (N,3) with N = grid_res^3.
    A_vals, B_vals, C_vals = np.meshgrid(grid_vals, grid_vals, grid_vals)
    combos = np.stack([A_vals.ravel(), B_vals.ravel(), C_vals.ravel()], axis=-1)  # shape (N,3)
    
    # For each candidate, compute candidate = a*E_basis[0] + b*E_basis[1] + c*E_basis[2] + E_basis[3].
    N = combos.shape[0]
    # Preallocate candidate matrices: shape (N, 3, 3)
    candidates = np.empty((N, 3, 3))
    for i in range(N):
        a, b, c = combos[i]
        candidates[i] = a*E_basis[0] + b*E_basis[1] + c*E_basis[2] + E_basis[3]
    # Enforce rank-2 constraint on each candidate by performing SVD
    for i in range(N):
        Ue, Se, Vte = la.svd(candidates[i])
        Se[2] = 0
        candidates[i] = Ue @ np.diag(Se) @ Vte

    # Prepare the 5 correspondences in homogeneous coordinates.
    pts1_5 = np.hstack((pts1, np.ones((5, 1)))).T  # shape (3,5)
    pts2_5 = np.hstack((pts2, np.ones((5, 1)))).T  # shape (3,5)

    # Vectorized error computation:
    # For each candidate in candidates (shape (N,3,3)), compute candidate @ pts1_5, shape (N,3,5)
    candidates_pts1 = candidates @ pts1_5  # shape (N,3,5)
    # For each candidate, compute residual for each of the 5 points: 
    # residuals = pt2.T dot (candidate @ pt1) for each point.
    # We can compute this with einsum:
    residuals = np.einsum('nij,jk->nik', candidates, pts1_5)  # shape (N,3,5)
    # Now, for each candidate, compute dot product with pts2_5 (shape (3,5)).
    # Compute per-candidate per-point residual: 
    res = np.einsum('ij,nij->ni', pts2_5, residuals)  # shape (N,5)
    total_errors = np.sum(np.abs(res), axis=1)  # shape (N,)

    best_index = np.argmin(total_errors)
    best_candidate = candidates[best_index]

    # Now, form the essential matrix from the candidate fundamental matrix by:
    E_est = K.T @ best_candidate @ K
    return E_est

#############################################
# RANSAC for the Five-Point Algorithm (Optimized)
#############################################
def ransac_five_point(pts1, pts2, K, threshold=1.0, max_iterations=1000, grid_res=10):
    """
    Run RANSAC on the five-point algorithm (using our optimized grid search)
    to robustly estimate the essential matrix.
    
    Args:
        pts1 (np.ndarray): Nx2 raw points from the first image.
        pts2 (np.ndarray): Nx2 raw points from the second image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        threshold (float): Threshold for epipolar residual error.
        max_iterations (int): Maximum number of RANSAC iterations.
        grid_res (int): Grid resolution to use in the optimized five-point algorithm.
    
    Returns:
        best_E (np.ndarray): Estimated 3x3 essential matrix.
        best_mask (np.ndarray): Boolean inlier mask.
        iterations (int): Number of iterations run.
    """
    num_points = pts1.shape[0]
    if num_points < 5:
        raise ValueError("At least 5 correspondences are required.")
        
    best_E = None
    best_mask = None
    best_inlier_count = 0

    for iteration in range(max_iterations):
        sample_indices = np.random.choice(num_points, 5, replace=False)
        pts1_sample = pts1[sample_indices]
        pts2_sample = pts2[sample_indices]
        try:
            candidate_E = five_point_essential_optimized(pts1_sample, pts2_sample, K, grid_res=grid_res)
        except Exception as e:
            continue  # Skip if any error occurs in candidate computation

        # Compute the error (absolute epipolar residual) for every correspondence.
        errors = []
        for i in range(num_points):
            pt1 = np.array([pts1[i, 0], pts1[i, 1], 1])
            pt2 = np.array([pts2[i, 0], pts2[i, 1], 1])
            err = abs(pt2.T @ candidate_E @ pt1)
            errors.append(err)
        errors = np.array(errors)
        mask = errors < threshold
        inlier_count = np.sum(mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_E = candidate_E
            best_mask = mask
        
        # Optional early break if high inlier ratio found.
        if best_inlier_count > 0.9 * num_points:
            break

    return best_E, best_mask, iteration + 1

#############################################
# Main Function: Compare Custom and OpenCV RANSAC
#############################################
def main():
    # Set up synthetic correspondences.
    np.random.seed(42)
    num_points = 50
    pts1 = np.random.uniform(low=0, high=640, size=(num_points, 2))
    
    # Define a known transformation (small rotation and translation)
    angle = np.deg2rad(5)
    R_true = np.array([[math.cos(angle), -math.sin(angle), 0],
                       [math.sin(angle),  math.cos(angle), 0],
                       [0,                0,               1]])
    t_true = np.array([[10], [5], [1]])
    
    # Camera intrinsic matrix K.
    K = np.array([[700, 0, 320],
                  [0, 700, 240],
                  [0,   0,   1]], dtype=float)
    
    # Create synthetic pts2 via the known transformation.
    pts2 = []
    for pt in pts1:
        pt_h = np.array([[pt[0]], [pt[1]], [1]])
        pt2_h = R_true @ pt_h + t_true
        pt2 = (pt2_h / pt2_h[2]).flatten()[:2]
        pts2.append(pt2)
    pts2 = np.array(pts2)
    # Add noise.
    pts2 += np.random.normal(scale=1.0, size=pts2.shape)
    
    # Run custom RANSAC five-point algorithm.
    E_custom, mask_custom, iters_custom = ransac_five_point(pts1, pts2, K, threshold=1.0, max_iterations=1000, grid_res=10)
    
    print("Custom five-point RANSAC estimated essential matrix:")
    print(E_custom)
    print("Custom inlier mask:")
    print(mask_custom)
    print("Custom iterations:", iters_custom)
    
    # Now use OpenCV's findEssentialMat.
    pts1_cv = pts1.astype(np.float32)
    pts2_cv = pts2.astype(np.float32)
    E_cv, mask_cv = cv2.findEssentialMat(pts1_cv, pts2_cv, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    print("\nOpenCV findEssentialMat estimated essential matrix:")
    print(E_cv)
    print("OpenCV inlier mask:")
    print(mask_cv.ravel())
    
    # Normalize both matrices (they are up to scale) and compare.
    E_custom_norm = E_custom / la.norm(E_custom)
    E_cv_norm = E_cv / la.norm(E_cv)
    diff = la.norm(E_custom_norm - E_cv_norm)
    print("\nFrobenius norm difference between normalized matrices:")
    print(diff)
    
if __name__ == "__main__":
    main()
