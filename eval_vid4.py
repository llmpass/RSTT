"""Evalute Space-Time Video Super-Resolution on Vid4 dataset.
"""
import os
import cv2
import glob
import torch
import lpips
import logging
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2ycbcr

from models import create_model
from utils import (mkdirs, parse_config, AverageMeter, structural_similarity, 
                   read_seq_images, index_generation, setup_logger, get_model_total_params)

def main():
    parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution Evaluation on Vid4 dataset')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    save_path = config['path']['save_path'] # os.path.join(output_dir, dataset)
    mkdirs(save_path)
    setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)
    model = create_model(config)
    model_params = get_model_total_params(model)

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Data: {} - {}'.format(config['dataset']['name'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    logger.info('Model parameters: {} M'.format(model_params))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    loss_fn_alex = lpips.LPIPS(net='alex') 
    LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], 'BIx4', '*')))

    PSNR = []
    PSNR_Y = []
    SSIM = []
    SSIM_Y = []
    LPIPS_Alexnet = []

    for LR_path in LR_paths:
        sub_save_path = os.path.join(save_path, LR_path.split('/')[-1])
        mkdirs(sub_save_path)

        tested_index = []

        GT_path = LR_path.replace('BIx4', 'GT')
        imgs_LR = read_seq_images(LR_path)
        imgs_LR = imgs_LR.astype(np.float32) / 255.
        imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()
        imgs_GT = read_seq_images(GT_path)
        
        indices_list = index_generation(config['dataset']['num_out_frames'], imgs_LR.shape[0])
        
        clips_PSNR = AverageMeter()
        clips_PSNR_Y = AverageMeter()
        clips_SSIM = AverageMeter()
        clips_SSIM_Y = AverageMeter()
        clips_LPIPS_Alexnet = AverageMeter()
        
        for indices in indices_list:

            inputs = imgs_LR[indices[::2]].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
            outputs = outputs.cpu().squeeze().clamp(0, 1).numpy()
            
            # PSNR, SSIM for each frame
            for idx, frame_idx in enumerate(indices):
                if frame_idx in tested_index:
                    continue
                tested_index.append(frame_idx)
                
                output = (outputs[idx].squeeze().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
                target = imgs_GT[frame_idx]
                output_y = rgb2ycbcr(output)[..., 0]
                target_y = rgb2ycbcr(target)[..., 0]
                psnr = peak_signal_noise_ratio(output, target)
                psnr_y = peak_signal_noise_ratio(output_y, target_y, data_range=255)
                ssim = structural_similarity(output, target)
                ssim_y = structural_similarity(output_y, target_y)
                # compute LPIPS: image should be RGB, IMPORTANT: normalized to [-1,1]
                img_tensor = torch.from_numpy(output/255.)[None].permute(0,3,1,2).float()*2-1.0 
                img_gt_tensor = torch.from_numpy(target/255.)[None].permute(0,3,1,2).float()*2-1.0
                lpips_val = loss_fn_alex(img_tensor, img_gt_tensor).item()
                cv2.imwrite(os.path.join(sub_save_path, '{:08d}.png'.format(frame_idx+1)), output[...,::-1])

                clips_PSNR.update(psnr)
                clips_PSNR_Y.update(psnr_y)
                clips_SSIM.update(ssim)
                clips_SSIM_Y.update(ssim_y)
                clips_LPIPS_Alexnet.update(lpips_val)

                msg = '{:3d} - PSNR: {:.6f} dB  PSNR-Y: {:.6f} dB ' \
                      'SSIM: {:.6f} SSIM-Y: {:.6f} ' \
                      'LPIPS: {:.6f}'.format(
                          frame_idx + 1, psnr, psnr_y, ssim, ssim_y, lpips_val
                      )
                logger.info(msg)

        msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
              'Average SSIM: {:.6f} SSIM-Y: {:.6f} ' \
              'Average LPIPS (Alexnet): {:.6f} for {} frames; '.format(
                  LR_path.split('/')[-1], clips_PSNR.average(), 
                  clips_PSNR_Y.average(), clips_SSIM.average(), 
                  clips_SSIM_Y.average(), clips_LPIPS_Alexnet.average(), clips_PSNR.count
              )
        logger.info(msg)
        PSNR.append(clips_PSNR.average())
        PSNR_Y.append(clips_PSNR_Y.average())
        SSIM.append(clips_SSIM.average())
        SSIM_Y.append(clips_SSIM_Y.average())
        LPIPS_Alexnet.append(clips_LPIPS_Alexnet.average())

    logger.info('################ Tidy Outputs ################')
    for path, psnr, psnr_y, ssim, ssim_y, lpips_alexnet in zip(LR_paths, PSNR, PSNR_Y, SSIM, SSIM_Y, LPIPS_Alexnet):
        msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
              'SSIM: {:.6f} dB SSIM-Y: {:.6f} dB LPIPS-Alexnet. '.format(
                  path.split('/')[-1], psnr, psnr_y, ssim, ssim_y, lpips_alexnet
              )
        logger.info(msg)
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(config['dataset']['name'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    msg = 'Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB SSIM: {:.6f} dB ' \
            'SSIM-Y: {:.6f} dB LPIPS: {:.6f} for {} clips.'.format(
              sum(PSNR) / len(PSNR), sum(PSNR_Y) / len(PSNR_Y), 
              sum(SSIM) / len(SSIM), sum(SSIM_Y) / len(SSIM_Y), sum(LPIPS_Alexnet) / len(LPIPS_Alexnet), len(PSNR)
          )
    logger.info(msg)

if __name__ == '__main__':
    main()