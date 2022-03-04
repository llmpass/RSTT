"""Separete Vimeo dataset into training and testing.
"""
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ProgressBar

def sep_vimeo(data_path, save_path, text_path):
    """Separate Vimeo90k dataset according to text_path.

    Args:
        data_path (str): Path to Vimeo dataset.
        save_path (str): Path to save training/test dataset.
        text_path (str): Path to training/test text file.
    """
    with open(text_path, 'r') as f:
        lines = f.readlines()
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    pbar = ProgressBar(len(lines))
    for l in lines:
        line = l.replace('\n','')
        pbar.update('Copy {}'.format('/'.join(line.split('/')[-2:])))
        src_dir = os.path.join(data_path, line)
        dst_dir = os.path.join(save_path, line)
        shutil.copytree(src_dir, dst_dir)
    print('Done')