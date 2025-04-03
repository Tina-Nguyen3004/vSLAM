from feature_extractor import *
from helpers import *

# -----------------------------
# Stereo functions (for initialization and update)
# -----------------------------
def match_features(dp1, dp2, ratio=0.75):
    """
    Match features between two images using BFMatcher and Lowe's ratio test.
    Args:
        dp1 (numpy.ndarray): Descriptors of the first image.
        dp2 (numpy.ndarray)): Descriptors of the second image.
        ratio (float): Ratio for Lowe's ratio test.
    Returns:
        matches: List of good matches.
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(dp1[1], dp2[1], k=2)
    # Apply Lowe's ratio test
    good_matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    return good_matches


def triangulate_points(kp_left, kp_right, matches, file_calib_path = "00/calib.txt"):
    """
    Triangulate points from matched keypoints.
    Args:
        kp_left (list): Keypoints from the left image.
        kp_right (list): Keypoints from the right image.
        matches (np.ndarray): List of good matches.
        file_path (str): Path to obtain the projection matrices
        
    Returns:
        points (np.ndarray): Nx3 array of triangulated 3D points.
    """
    # Obtain the points from the matched keypoints
    points_left = np.array([kp_left[m.trainIdx].pt for m in matches])
    points_right = np.array([kp_right[m.queryIdx].pt for m in matches])
    
    # Projection matrices for left and right cameras
    _, P_left = extract_intrinsic_parameters(file_calib_path, 'P0')
    _, P_right = extract_intrinsic_parameters(file_calib_path, 'P1')
    
    # Triangulate points 
    num_points = points_left.shape[0]
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        x1, y1 = points_left[i]
        x2, y2 = points_right[i]

        # Build the 4x4 matrix A
        A = np.zeros((4, 4))
        A[0] = x1 * P_left[2] - P_left[0]
        A[1] = y1 * P_left[2] - P_left[1]
        A[2] = x2 * P_right[2] - P_right[0]
        A[3] = y2 * P_right[2] - P_right[1]
        # Solve for X using SVD: A * X = 0
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Convert from homogeneous to Euclidean coordinates
        points[i] = X[:3]
    
    return points

def process_stereo(file_path, frame_index=0):
    """
    Processes a stereo frame for stereo VSLAM initialization or update.

    Args:
        file_path (str): Path to the dataset folder containing the stereo images and calibration file (e.g., "00").
        frame_index (int, optional): Index of the frame to initialize from (default is 0).

    Returns:
        tuple: A tuple containing:
            - img_left (np.ndarray): The left grayscale image from the stereo pair.
            - pts_3d (np.ndarray): An Nx3 array of triangulated 3D points.
            - prev_pts (np.ndarray): An array of 2D points (shape: [N, 1, 2]) from the left image for tracking.
    """
    # Load stereo pair 
    img_left, img_right = loadStereoPair(file_path, frame_index)
    # Extract features
    kp_left, dp_left = featureExtractor(img_left)
    kp_right, dp_right = featureExtractor(img_right)
    # Match features
    matches = match_features(dp_left, dp_right)
    # Triangulate points
    calib_file = os.path.join(file_path, "calib.txt")
    pts_3d = triangulate_points(kp_left, kp_right, matches, calib_file)
    
    # Prepare points for tracking: left images 
    prev_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    return img_left, pts_3d, prev_pts
    
# -----------------------------
# Tracking functions (for initialization and update)
# -----------------------------
def track_features(prev_img, curr_img, prev_pts):
    """
    Tracks feature points from a previous frame to the current frame using Lucas-Kanade optical flow.
    
    This function uses OpenCV's cv2.calcOpticalFlowPyrLK to estimate the new positions of a set of points (prev_pts)
    in the previous grayscale image (prev_img) when observed in the current grayscale image (curr_img). It returns
    the points in the previous image that were successfully tracked, along with their new positions in the current image.
    
    Args:
        prev_img (np.ndarray): The previous frame as a grayscale image.
        curr_img (np.ndarray): The current frame as a grayscale image.
        prev_pts (np.ndarray): An array of 2D points (shape: [N, 1, 2]) from the previous frame that need to be tracked.
        
    Returns:
        good_prev (np.ndarray): Array of points (shape: [M, 2]) from the previous image that were successfully tracked.
        good_curr (np.ndarray): Corresponding array of points (shape: [M, 2]) in the current image.
    """
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
    good_prev = prev_pts[status.flatten() == 1]
    good_curr = nextPts[status.flatten() == 1]
    return good_prev, good_curr

def draw_matches(img_left, kp_left, img_right, kp_right, matches, num_matches=20):
    """
    Draws matches between two images.
    
    Args:
        img_left (np.ndarray): Left image.
        kp_left (list): Keypoints from the left image.
        img_right (np.ndarray): Right image.
        kp_right (list): Keypoints from the right image.
        matches (list): List of good matches.
        num_matches (int): Number of matches to draw.
        
    Returns:
        img_matches (np.ndarray): Image with drawn matches.
    """
    # Draw only the first num_matches matches
    img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches[:num_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

if __name__ == "__main__":
    dataset_path = "00"
    frame_index = 0
    
    # Initialize the first 
    img_left_prev, pts_3d_prev, prev_pts = process_stereo(dataset_path, frame_index)
    
    kp_left_prev, dp_left_prev = featureExtractor(img_left_prev)
    img_left, img_right = loadStereoPair(dataset_path, frame_index)
    kp_left, dp_left = featureExtractor(img_left)
    kp_right, dp_right = featureExtractor(img_right)
    matches = match_features(dp_left, dp_right)
    img_matches = draw_matches(img_left_prev, kp_left_prev, img_right, kp_right, matches)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    
    # Loop over subsequent frames and track the points
    left_folder = os.path.join(dataset_path, "image_0")
    left_files = sort_files(os.listdir(left_folder))
    total_images = len(left_files)
    for i in range(1, total_images):
        cimg_left, cimg_right = loadStereoPair(dataset_path, i)
        good_prev, good_curr = track_features(img_left_prev, cimg_left, prev_pts)
        
        # Make a copy of the current image for debugging
        current_img_debug = cv2.cvtColor(cimg_left, cv2.COLOR_GRAY2BGR)
        for pt in good_curr:
            cv2.circle(current_img_debug, (int(pt[0][0]), int(pt[0][1])), 3, (0, 0, 255), -1)
        cv2.imshow("Tracked Points", current_img_debug)
        
        # Show matches between left and right images
        matches_debug = []
        for i, pt in enumerate(good_curr):
            if i < len(kp_left) and i < len(kp_right):  # Ensure indices are valid
                matches_debug.append(cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0))

        ckp_left, cdp_left = featureExtractor(cimg_left)
        ckp_right, cdp_right = featureExtractor(cimg_right)
        img_matches = draw_matches(cimg_left, ckp_left, cimg_right, ckp_right, matches_debug)
        cv2.imshow("Matches Current", img_matches)

        if frame_index % 5 == 0:
            img_left_prev, pts_3d_prev, prev_pts = process_stereo(dataset_path, i)
        else:
            img_left_prev = cimg_left.copy()
            prev_pts = good_curr.reshape(-1, 1, 2)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()