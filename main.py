#!/usr/bin/env python3
import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.transform import Rotation as R

# Import functions from your modules.
from feature_extractor import featureExtractor
from feature_matching import track_features  # Tracking (optical flow) for left images.
from helpers import sort_files, extract_intrinsic_parameters
from loop_closure import process_new_keyframe, is_keyframe, build_vocabulary, compute_bow_descriptor
from bundle_adjustment import run_bundle_adjustment_monocular, update_keyframes_from_ba
from pose_estimation import apply_ransac_eight_point, recover_pose

# ------------------ TRIANGULATION FUNCTION ------------------
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
    return points_3D.T  # Returns 4xN

# ------------------ HELPER FUNCTIONS ------------------
def match_features(kp1, des1, kp2, des2):
    """Match descriptors between two keyframes using BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_projection_matrix(K, pose):
    """
    Compute the camera projection matrix P = K * [R|t].
    
    Args:
        K (np.ndarray): Intrinsic matrix (3x3).
        pose (np.ndarray): 4x4 camera pose.
        
    Returns:
        np.ndarray: 3x4 projection matrix.
    """
    R_mat = pose[:3, :3]
    t = pose[:3, 3].reshape(3,1)
    Rt = np.hstack((R_mat, t))
    return K @ Rt

# ------------------ EXISTING FUNCTIONS ------------------
def update_pose_estimate(good_prev, good_curr, K, current_pose):
    if good_prev.shape[0] < 5 or good_curr.shape[0] < 5:
        print("Not enough correspondences for Essential Matrix estimation. Skipping pose update.")
        return current_pose
    E, mask = cv2.findEssentialMat(good_curr, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print("Essential matrix estimation failed. Keeping previous pose.")
        return current_pose
    R_mat, t = recover_pose(E, good_curr, good_prev, K)
    T_rel = np.eye(4)
    T_rel[:3, :3] = R_mat
    T_rel[:3, 3] = t.flatten()
    new_pose = current_pose @ T_rel
    return new_pose

def initialize_monocular(dataset_path, init_frame=0):
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    if init_frame >= len(image_files):
        raise IndexError("init_frame is out of range")
    img_path = os.path.join(left_folder, image_files[init_frame])
    img_left = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_left is None:
        raise FileNotFoundError("Left image not found at " + img_path)
    kp, _ = featureExtractor(img_left)
    if len(kp) == 0:
        raise ValueError("No keypoints detected in the initialization frame.")
    prev_pts = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)
    current_pose = np.eye(4)
    return img_left, prev_pts, current_pose

def display_trajectory(ax, trajectory):
    ax.clear()
    if trajectory:
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 0], traj_array[:, 2], 'b-', linewidth=2, label="Camera Trajectory")
        ax.scatter(traj_array[:, 0], traj_array[:, 2], c='red')
    ax.set_title("Real-Time Camera Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    plt.draw()

def display_pose_graph(pose_graph, pause_time=0.1):
    plt.figure("Pose Graph")
    plt.clf()
    pos = {}
    for node, data in pose_graph.nodes(data=True):
        pose = data.get('pose', np.eye(4))
        t = pose[:3, 3]
        pos[node] = (t[0], t[2])
    nx.draw(pose_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', font_weight='bold')
    plt.title("Keyframe Pose Graph")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.draw()
    plt.pause(pause_time)

# ------------------ MAIN PIPELINE ------------------
def main():
    # Enable interactive plotting.
    plt.ion()
    fig_traj, ax_traj = plt.subplots(num="Camera Trajectory")
    plt.figure("Pose Graph")
    
    # Define dataset folder.
    dataset_path = "00"
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    total_frames = len(image_files)
    print(f"Total frames: {total_frames}")
    
    # Load calibration and compute intrinsic matrix from left camera.
    calib_file = os.path.join(dataset_path, "calib.txt")
    K, _ = extract_intrinsic_parameters(calib_file, "P0")
    
    # Build vocabulary (using first 20 frames).
    descriptors_list = []
    num_train = min(20, total_frames)
    for i in range(num_train):
        img_path = os.path.join(left_folder, image_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, des = featureExtractor(img)
        if des is not None:
            descriptors_list.append(des)
    if len(descriptors_list) == 0:
        print("No descriptors for vocabulary training!")
        return
    print("Building vocabulary...")
    kmeans = build_vocabulary(descriptors_list, num_clusters=500)
    print("Vocabulary built.")
    
    # Initialize keyframe database, pose graph, landmark and observation databases.
    keyframe_db = []
    pose_graph = nx.Graph()
    landmarks_db = []       # Global 3D landmarks (from triangulation)
    observations_db = []    # List of observations: each {'camera_idx', 'point_idx', 'observation': [x,y]}
    last_keyframe_index = -100  # Force first frame as keyframe.
    keyframe_counter = 0
    loop_closure_constraints = []

    # Initialize monocular odometry.
    img_left_prev, prev_pts, current_pose = initialize_monocular(dataset_path, init_frame=0)
    trajectory = [current_pose[:3, 3].copy()]
    
    # Add the initial keyframe.
    kp, des = featureExtractor(img_left_prev)
    bow_descriptor = compute_bow_descriptor(des, kmeans)
    entry = {'id': keyframe_counter, 'bow': bow_descriptor, 'pose': current_pose, 'kp': kp, 'des': des}
    keyframe_db.append(entry)
    pose_graph.add_node(keyframe_counter, pose=current_pose)
    last_keyframe_index = 0
    keyframe_counter += 1
    
    reinit_interval = 3
    frames_since_reinit = 0
    
    # Main image processing loop.
    for frame_idx in range(1, round(total_frames/3)):
        img_path = os.path.join(left_folder, image_files[frame_idx])
        img_left = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_left is None:
            continue

        if frames_since_reinit >= reinit_interval:
            print(f"Reinitializing features at frame {frame_idx}.")
            kp, _ = featureExtractor(img_left)
            if len(kp) > 0:
                prev_pts = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)
            frames_since_reinit = 0
        else:
            good_prev, good_curr = track_features(img_left_prev, img_left, prev_pts)
            good_prev = good_prev.reshape(-1, 2)
            good_curr = good_curr.reshape(-1, 2)
            prev_pts = good_curr.reshape(-1, 1, 2)
            frames_since_reinit += 1

        # Update global pose.
        current_pose = update_pose_estimate(good_prev, good_curr, K, current_pose)
        trajectory.append(current_pose[:3, 3].copy())
        display_trajectory(ax_traj, trajectory)
        
        # Display tracked features.
        img_tracked = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        for pt in good_curr:
            cv2.circle(img_tracked, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.imshow("Tracked Points", img_tracked)
        cv2.waitKey(1)
        
        # Keyframe selection.
        if is_keyframe(frame_idx, last_keyframe_index, threshold=5):
            print(f"Frame {frame_idx} selected as keyframe.")
            candidate, rel_pose, (kp_curr, des_curr) = process_new_keyframe(img_left, keyframe_counter, keyframe_db, kmeans, current_pose)
            
            # Triangulate landmarks between the reference keyframe and the new keyframe.
            ref_kf = keyframe_db[-1]   # Use last keyframe as reference.
            kp_ref, des_ref = ref_kf['kp'], ref_kf['des']
            matches = match_features(kp_ref, des_ref, kp_curr, des_curr)
            print(f"Found {len(matches)} matches for triangulation.")
            if len(matches) >= 8:
                # Compute projection matrices.
                P_ref = compute_projection_matrix(K, ref_kf['pose'])
                P_curr = compute_projection_matrix(K, current_pose)
                # Extract matched point coordinates.
                pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])
                # Use your triangulation function.
                pts4D = triangulate_for_points(P_ref, P_curr, pts_ref, pts_curr)  # 4xN
                pts3D = (pts4D[:3, :] / pts4D[3, :]).T  # Convert to Nx3 array.
                print(f"Triangulated {pts3D.shape[0]} landmarks.")
                # Add each new landmark and its observations.
                for i in range(pts3D.shape[0]):
                    landmark_idx = len(landmarks_db)
                    landmarks_db.append(pts3D[i])
                    # Observation in reference keyframe.
                    observations_db.append({
                        'camera_idx': ref_kf['id'],
                        'point_idx': landmark_idx,
                        'observation': pts_ref[i]
                    })
                    # Observation in current keyframe.
                    observations_db.append({
                        'camera_idx': keyframe_counter,
                        'point_idx': landmark_idx,
                        'observation': pts_curr[i]
                    })
            else:
                print("Not enough matches for triangulation.")
            
            # Add the new keyframe.
            new_entry = {'id': keyframe_counter,
                         'bow': compute_bow_descriptor(des_curr, kmeans),
                         'pose': current_pose, 'kp': kp_curr, 'des': des_curr}
            keyframe_db.append(new_entry)
            pose_graph.add_node(keyframe_counter, pose=current_pose)
            if keyframe_counter > 0:
                pose_graph.add_edge(keyframe_counter - 1, keyframe_counter, measurement="odometry")
            if candidate is not None:
                R_loop, t_loop = rel_pose
                print(f"Loop closure detected: keyframe {keyframe_counter} matches keyframe {candidate['id']}")
                pose_graph.add_edge(candidate['id'], keyframe_counter, measurement="loop", rel_pose=rel_pose)
                loop_closure_constraints.append({
                    'current_id': keyframe_counter,
                    'candidate_id': candidate['id'],
                    'relative_R': R_loop,
                    'relative_t': t_loop
                })
            else:
                print(f"No loop closure detected for keyframe {keyframe_counter}.")
            last_keyframe_index = frame_idx
            keyframe_counter += 1
            display_pose_graph(pose_graph, pause_time=0.05)
        
        img_left_prev = img_left.copy()
    
    cv2.destroyAllWindows()
    print("Loop closure constraints collected:")
    for constraint in loop_closure_constraints:
        print(constraint)
    display_pose_graph(pose_graph, pause_time=1.0)
    plt.ioff()
    plt.show()
    
    # ----------------- BUNDLE ADJUSTMENT -----------------
    print("Running bundle adjustment for monocular VSLAM...")
    optimized_params = run_bundle_adjustment_monocular(
        initial_params=None,  # The BA module assembles initial parameters from camera poses and landmarks.
        n_cameras=len(keyframe_db),
        n_points=len(landmarks_db),
        camera_indices=np.array([obs['camera_idx'] for obs in observations_db]),
        point_indices=np.array([obs['point_idx'] for obs in observations_db]),
        observations=np.array([obs['observation'] for obs in observations_db]),
        K=K,
        max_iter=100
    )
    update_keyframes_from_ba(keyframe_db, optimized_params)
    n_cameras = len(keyframe_db)
    optimized_landmarks = optimized_params[n_cameras*6:].reshape(len(landmarks_db), 3)
    print("Optimized landmarks:")
    print(optimized_landmarks)
    display_pose_graph(pose_graph, pause_time=1.0)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
