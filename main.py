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

###############################################################################
# Monocular Pose Estimation Update using Essential Matrix
###############################################################################
def update_pose_estimate(good_prev, good_curr, K, current_pose):
    """
    Update camera pose by computing the Essential matrix from two sets of 2D points
    tracked in consecutive frames and then recovering the relative pose.
    """
    if good_prev.shape[0] < 5 or good_curr.shape[0] < 5:
        return current_pose

    E, mask = cv2.findEssentialMat(
        good_curr, good_prev, K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return current_pose

    R, t = recover_pose(E, good_curr, good_prev, K)
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.flatten()
    return current_pose @ T_rel


def initialize_monocular(dataset_path, init_frame=0):
    """
    Initialize monocular odometry using the left image only.
    """
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    if init_frame >= len(image_files):
        raise IndexError("init_frame is out of range")
    img_left = cv2.imread(
        os.path.join(left_folder, image_files[init_frame]),
        cv2.IMREAD_GRAYSCALE
    )
    if img_left is None:
        raise FileNotFoundError(f"Left image not found at {img_left}")

    kp, _ = featureExtractor(img_left)
    if not kp:
        raise ValueError("No keypoints detected in the initialization frame.")
    prev_pts = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)
    current_pose = np.eye(4)
    return img_left, prev_pts, current_pose


def display_trajectory(ax, trajectory):
    ax.clear()
    if trajectory:
        traj = np.array(trajectory)
        ax.plot(traj[:,0], traj[:,2], '-', linewidth=2, label="Camera Trajectory")
        ax.scatter(traj[:,0], traj[:,2], c='red')
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
        p = data.get('pose', np.eye(4))
        t = p[:3,3]
        pos[node] = (t[0], t[2])
    nx.draw(pose_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.title("Keyframe Pose Graph")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.draw()
    plt.pause(pause_time)


def main():
    plt.ion()
    fig_traj, ax_traj = plt.subplots(num="Camera Trajectory")

    dataset_path = "00"
    left_folder = os.path.join(dataset_path, "image_0")
    image_files = sort_files(os.listdir(left_folder))
    total_frames = len(image_files)

    calib_file = os.path.join(dataset_path, "calib.txt")
    K, _ = extract_intrinsic_parameters(calib_file, "P0")

    # Build vocabulary on first 20 frames
    descriptors_list = []
    for i in range(min(20, total_frames)):
        img = cv2.imread(os.path.join(left_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, des = featureExtractor(img)
        if des is not None:
            descriptors_list.append(des)
    kmeans = build_vocabulary(descriptors_list, num_clusters=500)

    keyframe_db = []
    pose_graph = nx.Graph()
    last_keyframe_index = -100
    keyframe_counter = 0
    loop_closure_constraints = []

    img_left_prev, prev_pts, current_pose = initialize_monocular(dataset_path)
    trajectory = [current_pose[:3,3].copy()]

    # Add first keyframe with frame_idx
    kp, des = featureExtractor(img_left_prev)
    bow = compute_bow_descriptor(des, kmeans)
    entry0 = {
        'id': 0,
        'frame_idx': 0,
        'bow': bow,
        'pose': current_pose,
        'kp': kp,
        'des': des
    }
    keyframe_db.append(entry0)
    pose_graph.add_node(0, pose=current_pose)
    keyframe_counter = 1
    last_keyframe_index = 0

    reinit_interval = 3
    frames_since_reinit = 0

    for frame_idx in range(1, total_frames):
        img_left = cv2.imread(os.path.join(left_folder, image_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if img_left is None:
            continue

        if frames_since_reinit >= reinit_interval:
            kp, _ = featureExtractor(img_left)
            if kp:
                prev_pts = np.float32([p.pt for p in kp]).reshape(-1,1,2)
            frames_since_reinit = 0
        else:
            good_prev, good_curr = track_features(img_left_prev, img_left, prev_pts)
            good_prev = good_prev.reshape(-1,2)
            good_curr = good_curr.reshape(-1,2)
            prev_pts = good_curr.reshape(-1,1,2)
            frames_since_reinit += 1

        current_pose = update_pose_estimate(good_prev, good_curr, K, current_pose)
        trajectory.append(current_pose[:3,3].copy())
        display_trajectory(ax_traj, trajectory)

        # show tracked points
        vis = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        for pt in good_curr:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)
        cv2.imshow("Tracked Points", vis)
        cv2.waitKey(1)

        # Keyframe selection
        if is_keyframe(frame_idx, last_keyframe_index, threshold=5):
            candidate, rel_pose, (kp, des) = process_new_keyframe(
                img_left, keyframe_counter, keyframe_db, kmeans, current_pose
            )
            # Append new keyframe entry with frame_idx
            bow = compute_bow_descriptor(des, kmeans)
            entry = {
                'id': keyframe_counter,
                'frame_idx': frame_idx,
                'bow': bow,
                'pose': current_pose,
                'kp': kp,
                'des': des
            }
            keyframe_db.append(entry)
            pose_graph.add_node(keyframe_counter, pose=current_pose)
            pose_graph.add_edge(keyframe_counter-1, keyframe_counter, measurement="odometry")

            if candidate is not None:
                cand_id = candidate['id']
                cand_frame = keyframe_db[cand_id]['frame_idx']
                curr_frame = frame_idx
                R_loop, t_loop = rel_pose
                print(f"🔄 Loop closure! frame {curr_frame} ↔ frame {cand_frame} "
                      f"(keyframes {keyframe_counter} ↔ {cand_id})")
                # Save constraint for reporting
                loop_closure_constraints.append({
                    'current_kf_id': keyframe_counter,
                    'current_frame': curr_frame,
                    'candidate_kf_id': cand_id,
                    'candidate_frame': cand_frame,
                    'relative_R': R_loop,
                    'relative_t': t_loop
                })
                # Optional: visualize the image pair
                img_cand = cv2.imread(os.path.join(left_folder, image_files[cand_frame]),
                                      cv2.IMREAD_GRAYSCALE)
                pair = np.hstack([img_left, img_cand])
                cv2.imshow("Loop Closure Pair", pair)
                cv2.waitKey(1)

            last_keyframe_index = frame_idx
            keyframe_counter += 1
            display_pose_graph(pose_graph, pause_time=0.05)

        img_left_prev = img_left.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    # Print all loop closures at end
    for lc in loop_closure_constraints:
        print(lc)

    display_pose_graph(pose_graph, pause_time=1.0)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
