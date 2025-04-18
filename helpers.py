import os
import numpy as np

def sort_files(file_list):
    """
    Sort file names by extracting the numeric part.
    
    Args:
        file_list (list): List of filenames (e.g., ['000000.png', '000001.png', ...]).
    
    Returns:
        Sorted list of filenames.
    """
    return sorted(file_list, key=lambda x: int(x.split('.')[0]))

def extract_intrinsic_parameters(file_calib_path = "00/calib.txt", camera = "P1"):
    """
    Extract intrinsic parameters from a file.
    
    Args:
        file_path (str): Path to the file containing intrinsic parameters.
        camera (str): Projection to use, for example, "P0" or "P1".
    
    Returns:
        K (numpy.ndarray): 3x3 Intrinsic matrix.
        P (numpy.ndarray): 3x4 Projection matrix.
    """
    with open(file_calib_path, 'r') as file:
        for line in file:
            if line.startswith(camera):
                # Remove the camera identifier and colon, then split into numbers
                numbers = line.strip().split()[1:]
                numbers = [float(num) for num in numbers]
                break
            
    P = np.array(numbers).reshape(3, 4)
    K = P[:, :3]
    return K, P

def draw_inlier_points(img, points, inliers_mask, color=(0, 255, 0), radius=5, thickness=2):
    """
    Draw circles on the image for inlier points.
    
    Args:
        img (np.ndarray): The image on which to draw.
        points (np.ndarray): Nx2 array of points (pixel coordinates).
        inliers_mask (np.ndarray): Boolean mask (N,) indicating which points are inliers.
        color (tuple): Color for the circles (B, G, R).
        radius (int): Radius of the circles.
        thickness (int): Thickness of the circle border.
    
    Returns:
        img_out (np.ndarray): The image with drawn circles.
    """
    img_out = img.copy()
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img_out
    for pt, inlier in zip(points, inliers_mask):
        if inlier:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img_out, (x, y), radius, color, thickness)
    return img_out

def compute_projection_matrix(K, pose):
    """
    Compute the camera projection matrix P = K * [R|t].
    """
    R_mat = pose[:3, :3]
    t = pose[:3, 3].reshape(3, 1)
    Rt = np.hstack((R_mat, t))
    return K @ Rt

def compute_similarity_transform(A, B, with_scale=False):
    assert A.shape == B.shape
    m = A.shape[0]
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.T @ BB / m
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = 1.0
    if with_scale:
        var_A = (AA**2).sum() / m
        s = (S.sum() / var_A)
    t = mu_B - s * R @ mu_A
    return s, R, t
    
if __name__ == "__main__":    
    K, P = extract_intrinsic_parameters("00/calib.txt")
    print(P)