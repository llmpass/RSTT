"""Evalute Space-Time Video Super-Resolution on Vimeo90k dataset.
"""
import os
import glob
import cv2
import argparse
import numpy as np
import torch
import logging
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2ycbcr

from models import create_model
from utils import (mkdirs, parse_config, AverageMeter, structural_similarity, 
                   read_seq_images, index_generation, setup_logger, get_model_total_params)

def main():
    parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution Evaluation on Vimeo90k dataset')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    save_path = config['path']['save_path'] 
    mkdirs(save_path)
    setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)
    model = create_model(config)
    model_params = get_model_total_params(model)

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Data: {} - {} - {}'.format(config['dataset']['name'], config['dataset']['mode'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    logger.info('Model parameters: {} M'.format(model_params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    # 왜 LR_paths가 비어있지?? => 밑 경로가 잘못돼있어서 그런거였음.
    # config['dataset']['dataset_root'] = "/home/gyubin/RSTT/to/vimeo/sequences_LR" #그래서 수정해줌
    config['dataset']['dataset_root'] = "./simple_test_dataset"
    

    LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], config['dataset']['mode']+'_test', '*')))
    print("LR_paths : ", LR_paths)
    print("config['dataset']['dataset_root'] : ", config['dataset']['dataset_root'])
    print("config['dataset']['mode'] : ", config['dataset']['mode'])

    print("config['dataset']['dataset_root'] : ", config['dataset']['dataset_root'])
    


    PSNR = []
    PSNR_Y = []
    SSIM = []
    SSIM_Y = []

    for LR_path in LR_paths: #현재 LR_path가 /simple_test_dataset/fast_test/frames
        sub_save_path = os.path.join(save_path, LR_path.split('/')[-1])
        mkdirs(sub_save_path)
        print("sub_save_path : ", sub_save_path)
        
        #여기서부터 vid4랑 코드 다 다름

        sub_LR_paths = sorted(glob.glob(os.path.join(LR_path, '*'))) #여기서 알아서 잘라주는데....?

        seq_PSNR = AverageMeter()
        seq_PSNR_Y = AverageMeter()
        seq_SSIM = AverageMeter()
        seq_SSIM_Y = AverageMeter()
        for sub_LR_path in sub_LR_paths:
            sub_sub_save_path = os.path.join(sub_save_path, sub_LR_path.split('/')[-1])
            mkdirs(sub_sub_save_path)

            tested_index = []

            sub_GT_path = sub_LR_path.replace('_LR', '')
            imgs_LR = read_seq_images(sub_LR_path) #여기서 에러가 남
            imgs_LR = imgs_LR.astype(np.float32) / 255.
            imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()
            imgs_GT = read_seq_images(sub_GT_path)
        
            indices_list = index_generation(config['dataset']['num_out_frames'], imgs_LR.shape[0])
        
            clips_PSNR = AverageMeter()
            clips_PSNR_Y = AverageMeter()
            clips_SSIM = AverageMeter()
            clips_SSIM_Y = AverageMeter()
            
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
                    # psnr = peak_signal_noise_ratio(output, target) #여기서 에러가 난다..?? same dimension이 아니라는 에러.. GT 이미지가 없네..
                    # psnr_y = peak_signal_noise_ratio(output_y, target_y, data_range=255)
                    # ssim = structural_similarity(output, target)
                    # ssim_y = structural_similarity(output_y, target_y)

                    cv2.imwrite(os.path.join(sub_save_path, '{:08d}.png'.format(frame_idx+1)), output[...,::-1])

                    # clips_PSNR.update(psnr)
                    # clips_PSNR_Y.update(psnr_y)
                    # clips_SSIM.update(ssim)
                    # clips_SSIM_Y.update(ssim_y)

                    # msg = '{:3d} - PSNR: {:.6f} dB  PSNR-Y: {:.6f} dB ' \
                    #     'SSIM: {:.6f} SSIM-Y: {:.6f}'.format(
                    #         frame_idx + 1, psnr, psnr_y, ssim, ssim_y
                    #     )
                    # logger.info(msg)

            # print("LR_path : ", LR_path) #뭐지..? 숫자로 돼있어야 하나?
            # sub_LR_path = 
            
    #         msg = 'Folder {}/{} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
    #             'Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} frames; '.format(
    #                 LR_path.split('/')[-1], sub_LR_path.split('/')[-1], 
    #                 clips_PSNR.average(), clips_PSNR_Y.average(), 
    #                 clips_SSIM.average(), clips_SSIM_Y.average(), 
    #                 clips_PSNR.count
    #             )
    #         logger.info(msg)
    #         seq_PSNR.update(clips_PSNR.average())
    #         seq_PSNR_Y.update(clips_PSNR_Y.average())
    #         seq_SSIM.update(clips_SSIM.average())
    #         seq_SSIM_Y.update(clips_SSIM_Y.average())

    #     msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
    #           'Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} clips; '.format(
    #                 LR_path.split('/')[-1], seq_PSNR.average(), 
    #                 seq_PSNR_Y.average(), seq_SSIM.average(), 
    #                 seq_SSIM_Y.average(), seq_PSNR.count
    #            )
    #     logger.info(msg)
    #     PSNR.append(seq_PSNR.average())
    #     PSNR_Y.append(seq_PSNR_Y.average())
    #     SSIM.append(seq_SSIM.average())
    #     SSIM_Y.append(seq_SSIM_Y.average())


    # logger.info('################ Tidy Outputs ################')
    # for path, psnr, psnr_y, ssim, ssim_y in zip(LR_paths, PSNR, PSNR_Y, SSIM, SSIM_Y):
    #     msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
    #           'SSIM: {:.6f} dB SSIM-Y: {:.6f} dB. '.format(
    #               path.split('/')[-1], psnr, psnr_y, ssim, ssim_y
    #           )
    #     logger.info(msg)
    # logger.info('################ Final Results ################')
    # logger.info('Data: {} - {} - {}'.format(config['dataset']['name'], config['dataset']['mode'], config['dataset']['dataset_root']))
    # logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    # msg = 'Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB SSIM: {:.6f} dB ' \
    #       'SSIM-Y: {:.6f} dB for {} clips.'.format(
    #           sum(PSNR) / len(PSNR), sum(PSNR_Y) / len(PSNR_Y), 
    #           sum(SSIM) / len(SSIM), sum(SSIM_Y) / len(SSIM_Y), len(PSNR)
    #       )
    # logger.info(msg)

if __name__ == '__main__':
    main()