import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cosine
from feature_extractor import *

# -----------------------------
# Vocabulary & BoW Functions
# -----------------------------
def build_vocabulary(descriptors_list, num_clusters=500):
    """
    Build a visual vocabulary using MiniBatchKMeans.
    
    Args:
        descriptors_list (list of np.ndarray): List of descriptors from training images.
        num_clusters (int): Number of visual words.
        
    Returns:
        kmeans (MiniBatchKMeans): Trained k-means model.
    """
    all_desc = np.vstack(descriptors_list)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_desc)
    return kmeans

def compute_bow_descriptor(descriptors, kmeans):
    """
    Compute a Bag-of-Words descriptor from the provided descriptors.
    
    Args:
        descriptors (np.ndarray): ORB descriptors (shape: [N, descriptor_dim]).
        kmeans (MiniBatchKMeans): Pre-trained vocabulary.
        
    Returns:
        bow (np.ndarray): Normalized histogram (shape: [num_clusters,]).
    """
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    bow = hist.astype(np.float32)
    bow /= (bow.sum() + 1e-6)
    return bow

# -----------------------------
# Loop Closure Functions
# -----------------------------
def detect_loop(new_bow, keyframe_db, similarity_threshold=0.7):
    """
    Detect a loop closure candidate by comparing BoW descriptors.
    
    Args:
        new_bow (np.ndarray): BoW descriptor of the new keyframe.
        keyframe_db (list): List of previous keyframe entries (each a dict with keys 'id', 'bow', etc.).
        similarity_threshold (float): Cosine similarity threshold.
    
    Returns:
        candidate (dict or None): Candidate keyframe if a match is found; otherwise, None.
    """
    best_similarity = 0.0
    candidate = None
    for entry in keyframe_db:
        db_bow = entry['bow']
        similarity = 1 - cosine(new_bow, db_bow)
        if similarity > similarity_threshold and similarity > best_similarity:
            best_similarity = similarity
            candidate = entry
    return candidate

def geometric_verification(kp1, des1, kp2, des2, ratio=0.75, reproj_thresh=3.0):
    """
    Verify a loop candidate using ORB feature matching and essential matrix estimation.
    
    Args:
        kp1, des1: Keypoints and descriptors from the current keyframe.
        kp2, des2: Keypoints and descriptors from the candidate keyframe.
        ratio (float): Lowe's ratio test threshold.
        reproj_thresh (float): Reprojection error threshold for RANSAC.
    
    Returns:
        relative_pose (tuple or None): (R, t) if verification passes; otherwise None.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    
    if len(good_matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=reproj_thresh)
    if E is None:
        return None
    inliers = mask.ravel().tolist()
    if np.sum(inliers) < 20:
        return None
    
    _, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2)
    return (R_est, t_est)

def detect_loop(new_bow, keyframe_db, current_keyframe_id, min_temporal_gap=5, similarity_threshold=0.8):
    """
    Detects a loop closure candidate by comparing BoW descriptors with
    keyframes that are sufficiently far apart temporally.
    
    Args:
        new_bow (np.ndarray): BoW descriptor of the new keyframe.
        keyframe_db (list): List of previous keyframe entries.
        current_keyframe_id (int): ID or frame index of the new keyframe.
        min_temporal_gap (int): Minimal difference in keyframe id or frame index to consider.
        similarity_threshold (float): Cosine similarity threshold.

    Returns:
        candidate (dict or None): Candidate keyframe entry if a match is found; otherwise, None.
    """
    best_similarity = 0.0
    candidate = None
    for entry in keyframe_db:
        # Ignore candidates that are too close in time (likely consecutive)
        if abs(current_keyframe_id - entry['id']) < min_temporal_gap:
            continue
        
        db_bow = entry['bow']
        similarity = 1 - cosine(new_bow, db_bow)
        if similarity > similarity_threshold and similarity > best_similarity:
            best_similarity = similarity
            candidate = entry
    return candidate

def process_new_keyframe(img, keyframe_id, keyframe_db, kmeans, current_pose):
    """
    Process a new keyframe for loop closure. Extract features, compute BoW descriptor,
    perform loop detection and (if necessary) geometric verification.
    
    Args:
        img (np.ndarray): Grayscale image of the new keyframe.
        keyframe_id (int): Unique identifier for the keyframe.
        keyframe_db (list): Database of prior keyframes.
        kmeans (MiniBatchKMeans): Pre-trained vocabulary.
        current_pose (np.ndarray): Current estimated pose (6-parameter vector) for the keyframe.
    
    Returns:
        loop_candidate (dict or None): Information about the loop closure if detected;
                                       else, None.
        kp (list): ORB keypoints for the current keyframe.
        des (np.ndarray): ORB descriptors for the current keyframe.
    """
    kp, des = featureExtractor(img)
    if des is None or len(des) == 0:
        return None, None, None
    bow_descriptor = compute_bow_descriptor(des, kmeans)
    # Pass the keyframe_id to detect_loop to enable temporal filtering.
    candidate = detect_loop(bow_descriptor, keyframe_db, keyframe_id)
    
    if candidate is not None:
        relative_pose = geometric_verification(kp, des, candidate['kp'], candidate['des'])
        if relative_pose is not None:
            print(f"Loop closure detected between keyframe {keyframe_id} and keyframe {candidate['id']}.")
            return candidate, relative_pose, (kp, des)
    
    # If no valid loop closure is detected, add this keyframe to the database.
    keyframe_entry = {'id': keyframe_id, 'bow': bow_descriptor, 'pose': current_pose, 'kp': kp, 'des': des}
    keyframe_db.append(keyframe_entry)
    return None, None, (kp, des)


# -----------------------------
# Keyframe Selection Abstraction
# -----------------------------
def is_keyframe(frame_index, last_keyframe_index, threshold=5):
    """
    Simple keyframe selection based on frame interval.
    
    Args:
        frame_index (int): Current frame index.
        last_keyframe_index (int): Index of the last keyframe.
        threshold (int): Minimal number of frames between keyframes.
        
    Returns:
        bool: True if the current frame is to be considered a keyframe.
    """
    return (frame_index - last_keyframe_index) >= threshold

# -----------------------------
# Main Loop Closure Pipeline
# -----------------------------
def main_loop_closure(dataset_path):
    """
    A complete pipeline demonstrating loop closure without assuming a fixed 'keyframes' folder.
    This function loops through all images in the dataset folder and uses a keyframe selection
    function to decide which frames are keyframes.
    
    Args:
        dataset_path (str): Path to a folder containing sequence images.
    """
    # List all image files in the dataset path (assume they are sorted appropriately)
    image_files = sorted(os.listdir(dataset_path))
    
    # For demo, we'll build a vocabulary from the first 20 images.
    descriptors_list = []
    num_train = min(20, len(image_files))
    for i in range(num_train):
        img = cv2.imread(os.path.join(dataset_path, image_files[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, des = featureExtractor(img)
        if des is not None:
            descriptors_list.append(des)
    
    if len(descriptors_list) == 0:
        print("No descriptors available for vocabulary training!")
        return
    
    print("Building vocabulary...")
    kmeans = build_vocabulary(descriptors_list, num_clusters=500)
    print("Vocabulary built.")
    
    # Initialize keyframe database and keyframe selection variables.
    keyframe_db = []
    last_keyframe_index = -100  # ensure first frame becomes keyframe
    keyframe_counter = 0
    loop_closure_constraints = []  # Store loop closure constraints (for pose graph)
    
    # In a real system, current_pose comes from your tracking module.
    # Here we'll assume identity (zero) for simplicity.
    current_pose = np.zeros(6)
    
    # Loop through all images.
    for idx, file in enumerate(image_files):
        img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Determine if this frame qualifies as a keyframe.
        if is_keyframe(idx, last_keyframe_index, threshold=5):
            print(f"Processing frame {idx} as keyframe.")
            candidate, rel_pose, (kp, des) = process_new_keyframe(img, keyframe_counter, keyframe_db, kmeans, current_pose)
            if candidate is not None:
                # A loop closure is detected.
                R_loop, t_loop = rel_pose
                print(f"Loop closure: current keyframe {keyframe_counter} matches keyframe {candidate['id']}")
                loop_closure_constraints.append({
                    'current_id': keyframe_counter,
                    'candidate_id': candidate['id'],
                    'relative_R': R_loop,
                    'relative_t': t_loop
                })
            else:
                print(f"No loop detected for keyframe {keyframe_counter}. Adding to database.")
            
            last_keyframe_index = idx
            keyframe_counter += 1
        else:
            print(f"Frame {idx} skipped (not a keyframe).")
        
        # Display the current frame with detected keypoints (if keyframe)
        if idx - last_keyframe_index < 1:  # if keyframe, show drawn keypoints
            img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            cv2.imshow("Keyframe", img_kp)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    
    print("Loop closure constraints collected:")
    for constraint in loop_closure_constraints:
        print(constraint)

if __name__ == "__main__":
    # Replace this path with the directory where your sequence images reside.
    dataset_path = "00/image_0"  # Example path
    main_loop_closure(dataset_path)
