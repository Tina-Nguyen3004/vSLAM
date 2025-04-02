import cv2
import os
import numpy as np
from helpers import sort_files

def featureExtractor(img):
    """
    Extracts keypoints and descriptors from a grayscale image using ORB.
    
    Args:
        img (numpy.ndarray): Grayscale image.
    
    Returns:
        kp (list): List of keypoints.
        des (numpy.ndarray): Descriptor array.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()
    # Detect keypoints and compute descriptors
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def loadStereoPair(dataset_path, frame_index):
    """
    Loads a stereo pair of images (left and right) for a given frame index from the KITTI dataset.
    
    Args:
        dataset_path (str): Path to the KITTI sequence (e.g., "dataset/sequences/00").
        frame_index (int): Frame index to load.
    
    Returns:
        img_left (numpy.ndarray): Left image in grayscale.
        img_right (numpy.ndarray): Right image in grayscale.
    """
    left_folder = os.path.join(dataset_path, "image_0")
    right_folder = os.path.join(dataset_path, "image_1")
    
    # Get sorted list of image files
    left_files = sort_files(os.listdir(left_folder))
    right_files = sort_files(os.listdir(right_folder))
    
    if frame_index >= len(left_files) or frame_index >= len(right_files):
        raise IndexError("Frame index out of range.")
    
    # Load left and right images
    left_path = os.path.join(left_folder, left_files[frame_index])
    right_path = os.path.join(right_folder, right_files[frame_index])
    
    # Load images 
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if img_left is None or img_right is None:
        raise FileNotFoundError("Image files not found.")
    
    return img_left, img_right
    
if __name__ == "__main__":
    dataset_path = "00"
    frame_index = 0
    
    # Get the total images in the frame
    left_folder = os.path.join(dataset_path, "image_0")
    left_files = os.listdir(left_folder)
    total_images = len(left_files)
    
    for frame_index in range(total_images):
        img_left, img_right = loadStereoPair(dataset_path, frame_index)
        kp_left, des_left = featureExtractor(img_left)
        kp_right, des_right = featureExtractor(img_right)
        
        img_left_kp = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0), flags=0)
        img_right_kp = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0), flags=0)
        
        cv2.imshow("Left Image Keypoints", img_left_kp)
        cv2.imshow("Right Image Keypoints", img_right_kp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()