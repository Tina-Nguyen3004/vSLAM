#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# Import functions from your modules.
from feature_extractor import featureExtractor
from feature_matching import *  
from helpers import *
from loop_closure import *
from bundle import local_bundle_adjustment  # Your vectorized BA function
from pose_estimation import *

###############################################################################
# Monocular Pose Estimation Update using Essential Matrix
###############################################################################
def update_pose_estimate(good_prev, good_curr, K, current_pose):
    # If there are not enough correspondences, keep the current pose.
    if good_prev.shape[0] < 5 or good_curr.shape[0] < 5:
        return current_pose

    E, mask = cv2.findEssentialMat(good_curr, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return current_pose

    R, t = recover_pose(E, good_curr, good_prev, K)
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
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

def display_trajectory(ax, traj_raw, traj_ba):
    """
    Plot the camera trajectory with raw odometry poses and BA-refined poses.
    """
    ax.clear()
    if traj_raw:
        raw_arr = np.array(traj_raw)
        ax.plot(raw_arr[:, 0], raw_arr[:, 2], 'r-', linewidth=2, label="Raw Odometry")
    if traj_ba:
        ba_arr = np.array(traj_ba)
        ax.plot(ba_arr[:, 0], ba_arr[:, 2], 'b-', linewidth=2, label="BA Corrected")
    ax.set_title("Camera Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    plt.draw()

def gather_observations(recent_keyframes):
    """
    Dummy implementation for gathering observations from recent keyframes.
    You should replace this with your own logic.
    
    Returns a dictionary:
      - 'camera_indices': array of camera indices (one per observation).
      - 'point_indices': array of indices associating each observation to a landmark.
      - 'observations': array of observed 2D keypoints.
      - 'points_3d': initial estimates of 3D points.
    """
    n_keyframes = len(recent_keyframes)
    camera_indices = []
    point_indices = []
    observations = []
    for i in range(n_keyframes):
        # For simplicity, assume each keyframe observes two dummy points.
        camera_indices.extend([i, i])
        point_indices.extend([0, 1])
        # Use dummy observed coordinates shifted by the keyframe index.
        observations.append([320 + i, 240 + i])
        observations.append([330 + i, 245 + i])
    observations = np.array(observations, dtype=np.float32)
    points_3d = np.array([
        [1.0, 2.0, 10.0],
        [1.5, 1.8, 9.5]
    ], dtype=np.float32)
    return {
        'camera_indices': np.array(camera_indices, dtype=np.int32),
        'point_indices': np.array(point_indices, dtype=np.int32),
        'observations': observations,
        'points_3d': points_3d
    }

def main():
    plt.ion()
    fig_traj, ax_traj = plt.subplots(num="Camera Trajectory")
    
    dataset_path = "00"  # Directory with the KITTI images (e.g., "00/image_0")
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    total_frames = len(image_files)
    
    calib_file = os.path.join(dataset_path, "calib.txt")
    K, _ = extract_intrinsic_parameters(calib_file, "P0")
    
    # Build vocabulary from the first 20 frames.
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
    
    # Initialize keyframe database.
    keyframe_db = []
    last_keyframe_index = -100
    keyframe_counter = 0
    traj_raw = []  # Trajectory from raw odometry
    traj_ba = []   # Trajectory updated by BA
    
    img_left_prev, prev_pts, current_pose = initialize_monocular(dataset_path, init_frame=0)
    traj_raw.append(current_pose[:3, 3].copy())
    traj_ba.append(current_pose[:3, 3].copy())
    
    # Add the initial keyframe.
    kp, des = featureExtractor(img_left_prev)
    bow_descriptor = compute_bow_descriptor(des, kmeans)
    entry = {
        'id': keyframe_counter,
        'bow': bow_descriptor,
        'pose': current_pose.copy(),
        'kp': kp,
        'des': des
    }
    keyframe_db.append(entry)
    last_keyframe_index = 0
    keyframe_counter += 1
    
    reinit_interval = 3
    frames_since_reinit = 0
    
    for frame_idx in range(1, total_frames):
        img_path = os.path.join(left_folder, image_files[frame_idx])
        img_left = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_left is None:
            continue
        
        if frames_since_reinit >= reinit_interval:
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
        
        # Update pose using raw odometry.
        current_pose = update_pose_estimate(good_prev, good_curr, K, current_pose)
        traj_raw.append(current_pose[:3, 3].copy())
        
        # Optionally, display tracked features.
        img_tracked = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        for pt in good_curr:
            cv2.circle(img_tracked, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.imshow("Tracked Points", img_tracked)
        cv2.waitKey(1)
        
        # Keyframe selection condition.
        if is_keyframe(frame_idx, last_keyframe_index, threshold=2):
            # Process and add the new keyframe.
            candidate, rel_pose, (kp, des) = process_new_keyframe(
                img_left, keyframe_counter, keyframe_db, kmeans, current_pose)
            
            new_entry = {
                'id': keyframe_counter,
                'bow': compute_bow_descriptor(des, kmeans),
                'pose': current_pose.copy(),
                'kp': kp,
                'des': des
            }
            keyframe_db.append(new_entry)
            last_keyframe_index = frame_idx
            keyframe_counter += 1
            
            # Run BA if there are enough keyframes.
            window_size = 5
            if len(keyframe_db) >= window_size:
                recent_keyframes = keyframe_db[-window_size:]
                observations_data = gather_observations(recent_keyframes)
                optimized_poses = local_bundle_adjustment(recent_keyframes, observations_data, K)
                # Update keyframe poses with optimized values.
                for entry in recent_keyframes:
                    kf_id = entry['id']
                    if kf_id in optimized_poses:
                        entry['pose'] = optimized_poses[kf_id]
                # For display, update the BA trajectory using the last (most recent) keyframe.
                traj_ba.append(recent_keyframes[-1]['pose'][:3, 3].copy())
            else:
                # If BA has not run, simply update BA trajectory with current pose.
                traj_ba.append(current_pose[:3, 3].copy())
            
            print(f"Added keyframe {keyframe_counter - 1}")
        
        # Update previous image.
        img_left_prev = img_left.copy()
        display_trajectory(ax_traj, traj_raw, traj_ba)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
