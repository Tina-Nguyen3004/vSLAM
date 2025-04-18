#!/usr/bin/env python3
import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn.cluster import MiniBatchKMeans

# Import functions from your modules.
from feature_extractor import featureExtractor
from feature_matching import *  
from helpers import *
from loop_closure import *
from bundle_adjustment import *
from pose_estimation import *


def update_pose_estimate(good_prev, good_curr, K, current_pose):
    """
    Update camera pose by computing the Essential matrix from two sets of 2D points
    tracked in consecutive frames and then recovering the relative pose.
    
    Args:
        good_prev (np.ndarray): Tracked 2D points in the previous frame (shape: [N,2]).
        good_curr (np.ndarray): Tracked 2D points in the current frame (shape: [N,2]).
        K (np.ndarray): Camera intrinsic matrix (3x3) from the left projection.
        current_pose (np.ndarray): Current global pose as a 4x4 transformation matrix.
        
    Returns:
        np.ndarray: Updated 4x4 transformation matrix.
    """
    # Check if enough correspondences exist.
    if good_prev.shape[0] < 5 or good_curr.shape[0] < 5:
        return current_pose

    # Estimate the essential matrix using RANSAC.
    
    E, mask = cv2.findEssentialMat(good_curr, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return current_pose

    # Recover relative pose: rotation R and translation t.
    R, t = recover_pose(E, good_curr, good_prev, K)

    # Build the relative transformation matrix (4x4).
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.flatten()

    # Update global pose by composing the previous pose with the relative transformation.
    new_pose = current_pose @ T_rel
    return new_pose

def initialize_monocular(dataset_path, init_frame=0):
    """
    Initialize monocular odometry using the left image only.
    
    Args:
        dataset_path (str): Path to the KITTI sequence folder ("00").
        init_frame (int): Frame index to initialize from.
        
    Returns:
        img_left (np.ndarray): Grayscale left image.
        prev_pts (np.ndarray): 2D points (for tracking) from the left image.
        current_pose (np.ndarray): Initial global pose as a 4x4 transformation (identity).
    """
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    if init_frame >= len(image_files):
        raise IndexError("init_frame is out of range")
    img_path = os.path.join(left_folder, image_files[init_frame])
    img_left = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_left is None:
        raise FileNotFoundError("Left image not found at " + img_path)
    
    # Extract features and convert keypoints into an array for tracking.
    kp, _ = featureExtractor(img_left)
    if len(kp) == 0:
        raise ValueError("No keypoints detected in the initialization frame.")
    prev_pts = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)

    # Initial global pose is identity (4x4 homogeneous transformation).
    current_pose = np.eye(4)
    return img_left, prev_pts, current_pose

def display_trajectory(ax, trajectory):
    """
    Update the trajectory plot with the camera positions (top–down view using x and z).
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot the trajectory.
        trajectory (list of np.ndarray): List of camera translations (3,).
    """
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
    """
    Display or update the keyframe pose graph.
    
    Args:
        pose_graph (networkx.Graph): Graph with nodes containing a 'pose' attribute.
        pause_time (float): Pause duration to update the figure.
    """
    plt.figure("Pose Graph")
    plt.clf()
    pos = {}
    for node, data in pose_graph.nodes(data=True):
        # For a top-down view, use the x and z translation components.
        pose = data.get('pose', np.eye(4))
        t = pose[:3, 3]
        pos[node] = (t[0], t[2])
    nx.draw(pose_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.title("Keyframe Pose Graph")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.draw()
    plt.pause(pause_time)

def main():
    # Enable interactive plotting.
    plt.ion()
    fig_traj, ax_traj = plt.subplots(num="Camera Trajectory")
    
    # Define dataset folder.
    dataset_path = "00"
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    total_frames = len(image_files)
    
    # Load calibration and compute intrinsic matrix from left camera ("P0").
    calib_file = os.path.join(dataset_path, "calib.txt")
    K, _ = extract_intrinsic_parameters(calib_file, "P0")
    
    # (Optional) Build vocabulary from the first 20 frames for keyframe-based processing.
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
        return
    kmeans = build_vocabulary(descriptors_list, num_clusters=500)
    
    # Initialize keyframe database and pose graph.
    keyframe_db = []
    pose_graph = nx.Graph()
    last_keyframe_index = -100  # Force first frame as keyframe.
    keyframe_counter = 0
    loop_closure_constraints = []  # For storing loop closure info (if any).
    
    # Initialize monocular odometry.
    img_left_prev, prev_pts, current_pose = initialize_monocular(dataset_path, init_frame=0)
    
    # Save the initial camera translation (for trajectory visualization).
    trajectory = [current_pose[:3, 3].copy()]
    
    # Add initial keyframe to keyframe database and pose graph.
    kp, des = featureExtractor(img_left_prev)
    bow_descriptor = compute_bow_descriptor(des, kmeans)
    entry = {'id': keyframe_counter, 'bow': bow_descriptor, 'pose': current_pose, 'kp': kp, 'des': des}
    keyframe_db.append(entry)
    pose_graph.add_node(keyframe_counter, pose=current_pose)
    last_keyframe_index = 0
    keyframe_counter += 1
    
    # For feature reinitialization: re-extract features every 3 frames.
    reinit_interval = 3
    frames_since_reinit = 0
    
    # Main processing loop.
    for frame_idx in range(1, total_frames):
        # Load the current left image.
        img_path = os.path.join(left_folder, image_files[frame_idx])
        img_left = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_left is None:
            continue

        # Reinitialize features every reinit_interval frames.
        if frames_since_reinit >= reinit_interval:
            kp, _ = featureExtractor(img_left)
            if len(kp) > 0:
                prev_pts = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)
            frames_since_reinit = 0
        else:
            # Track features from previous left image using optical flow.
            good_prev, good_curr = track_features(img_left_prev, img_left, prev_pts)
            # Ensure the points are reshaped to (N,2)
            good_prev = good_prev.reshape(-1, 2)
            good_curr = good_curr.reshape(-1, 2)
            prev_pts = good_curr.reshape(-1, 1, 2)
            frames_since_reinit += 1

        # Update the global pose using the essential matrix.
        current_pose = update_pose_estimate(good_prev, good_curr, K, current_pose)
        
        # Update trajectory for visualization.
        trajectory.append(current_pose[:3, 3].copy())
        display_trajectory(ax_traj, trajectory)
        
        # Optionally, display tracked features.
        img_tracked = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        for pt in good_curr:
            cv2.circle(img_tracked, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.imshow("Tracked Points", img_tracked)
        cv2.waitKey(1)
        
        # Keyframe selection: here we select every 5th frame as a keyframe.
        keyframe_threshold = 5
        if is_keyframe(frame_idx, last_keyframe_index, threshold=keyframe_threshold):
            candidate, rel_pose, (kp, des) = process_new_keyframe(
                img_left, keyframe_counter, keyframe_db, kmeans, current_pose)
            # Add new keyframe to the pose graph.
            pose_graph.add_node(keyframe_counter, pose=current_pose)
            if keyframe_counter > 0:
                pose_graph.add_edge(keyframe_counter - 1, keyframe_counter, measurement="odometry")
            if candidate is not None:
                R_loop, t_loop = rel_pose
                old_match_keyframe = candidate['id']
                old_match_idx = old_match_keyframe * keyframe_threshold
                print(f"Loop closure detected: keyframe {keyframe_counter} with frame id {frame_idx} matches keyframe {old_match_keyframe} with frame id {old_match_idx}")
                # draw the pair of keyframes with frame ids
                img_match = cv2.imread(os.path.join(left_folder, image_files[old_match_idx]), cv2.IMREAD_GRAYSCALE)
                pair = np.hstack((img_left, img_match))
                window_name = f"Loop Closure Pair: frame ids {frame_idx} - {old_match_idx}"
                cv2.imshow(window_name, pair)
                cv2.waitKey(1)
                pose_graph.add_edge(candidate['id'], keyframe_counter, measurement="loop", rel_pose=rel_pose)
                loop_closure_constraints.append({
                    'current_id': keyframe_counter,
                    'candidate_id': candidate['id'],
                    'relative_R': R_loop,
                    'relative_t': t_loop,
                    'frame_idx': frame_idx
                })
            last_keyframe_index = frame_idx
            keyframe_counter += 1
            
            # Update the keyframe pose graph display.
            display_pose_graph(pose_graph, pause_time=0.05)
        
        # Update the previous image.
        img_left_prev = img_left.copy()
        
        # quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    for constraint in loop_closure_constraints:
        print(constraint)

    traj_array = np.array(trajectory)           # shape (N,3)
    np.savetxt(
        "camera_trajectory.csv",
        traj_array[:, [0,2]],
        delimiter=",",
        header="x,z",
        comments=""
    )

    fig_traj.savefig("camera_trajectory.png", dpi=300)

    plt.ioff()
    plt.show()

    display_pose_graph(pose_graph, pause_time=1.0)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()