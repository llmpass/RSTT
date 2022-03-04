"""Utilities for evaluation.
"""
import os
import glob
import numpy as np
import cv2 

def read_image(path):
    """Read image from the given path.

    Args:
        path (str): The path of the image.

    Returns:
        array: RGB image.
    """
    # RGB
    img = cv2.imread(path)[:, :, ::-1]
    return img

def read_seq_images(path):
    """Read a sequence of images.

    Args:
        path (str): The path of the image sequence.

    Returns:
        array: (N, H, W, C) RGB images.
    """
    imgs_path = sorted(glob.glob(os.path.join(path, '*')))
    imgs = [read_image(img_path) for img_path in imgs_path]
    imgs = np.stack(imgs, axis=0)
    return imgs

def index_generation(num_output_frames, num_GT):
    """Generate index list for evaluation. 
    Each list contains num_output_frames indices.

    Args:
        num_output_frames (int): Number of output frames.
        num_GT (int): Number of ground truth.

    Returns:
        list[list[int]]: A list of indices list for testing.
    """

    indices_list = []
    right = num_output_frames
    while (right <= num_GT):
        indices = list(range(right - num_output_frames, right))
        indices_list.append(indices)
        right += num_output_frames - 1
    
    # Check if the last frame is included
    if right - num_output_frames < num_GT - 1:
        indices = list(range(num_GT - num_output_frames, num_GT))
        indices_list.append(indices)      
    return indices_list
