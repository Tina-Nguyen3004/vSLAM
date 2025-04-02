import os

def sort_files(file_list):
    """
    Sort file names by extracting the numeric part.
    
    Args:
        file_list (list): List of filenames (e.g., ['000000.png', '000001.png', ...]).
    
    Returns:
        Sorted list of filenames.
    """
    return sorted(file_list, key=lambda x: int(x.split('.')[0]))