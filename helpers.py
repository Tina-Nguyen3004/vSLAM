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
    
if __name__ == "__main__":    
    K, P = extract_intrinsic_parameters("00/calib.txt")
    print(P)