"""Generate low resolution images for Vimeo dataset.
"""
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imresize_np


def generate_LR(data_path, save_path, up_scale):
    """Generate low-resolution image using bicubic interpolation

    Args:
        data_path (str): Path to high-resolution images.
        save_path (str): Path to save low-resolution images.
        up_scale (int): Upscale factor.
    """
    if not os.path.isdir(data_path):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    filenames = [f for f in os.listdir(data_path) if f.endswith('.png')]

    for filename in filenames:
        # Read image
        image = cv2.imread(os.path.join(data_path, filename))

        # Resize image
        width = int(np.floor(image.shape[1] / up_scale))
        height = int(np.floor(image.shape[0] / up_scale))
        image_HR = image[0:up_scale * height, 0:up_scale * width, :]
        image_LR = imresize_np(image_HR, 1 / up_scale, True)

        cv2.imwrite(os.path.join(save_path, filename), image_LR)