"""Compute FPS on Vid4 dataset.
"""
import os
import argparse
import glob
import time
import numpy as np
import torch

from utils import parse_config, read_seq_images
from models import create_model

def main():
    parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution FPS computation on Vid4 dataset')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], 'BIx4', '*')))
    imgs_LR = read_seq_images(LR_paths[0])
    imgs_LR = imgs_LR.astype(np.float32) / 255.
    imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()


    model = create_model(config)
    device = torch.device('cuda')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    inputs = imgs_LR[10:14].unsqueeze(0).to(device)
    torch.cuda.synchronize()
    start = time.time()

    n = 100
    for i in range(n):
        with torch.no_grad():
            outputs = model(inputs)

    torch.cuda.synchronize()
    end = time.time()
    print('fps =', n*7.0/(end-start))

if __name__ == '__main__':
    main()