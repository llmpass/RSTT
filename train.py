import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

from data import create_dataloader, create_dataset
from tools import Trainer
from utils import (parse_config, dict2str, dict_to_nonedict, setup_logger, 
                   mkdir_and_rename, mkdirs, set_random_seed, check_resume)


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    config = parse_config(args.config, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        config['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        config['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # Loading resume state if exists
    if config['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(config['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(config, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            mkdir_and_rename(config['path']['experiments_root'])  # rename experiment folder if exists
            mkdirs((path for key, path in config['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        setup_logger(
            'base', config['path']['log'], 'train_' + config['name'], 
            level=logging.INFO, screen=True, tofile=True
        )
        logger = logging.getLogger('base')
        logger.info(dict2str(config))

        # tensorboard logger
        if config['use_tb_logger'] and 'debug' not in config['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + config['name'])
    else:
        setup_logger('base', config['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    config = dict_to_nonedict(config)

    # random seed
    seed = config['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_set = create_dataset(config['dataset'])
    train_size = int(math.ceil(len(train_set) / config['dataset']['batch_size']))
    total_iters = int(config['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    if config['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
    else:
        train_sampler = None
    train_loader = create_dataloader(train_set, config['dataset'], config, train_sampler)
    if rank <= 0:
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))
    assert train_loader is not None

    trainer = Trainer(config)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        trainer.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if config['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            trainer.train_one_sample(train_data, current_step)

            # log
            if current_step % config['logger']['print_freq'] == 0:
                logs = trainer.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in trainer.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')>'
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if config['use_tb_logger'] and 'debug' not in config['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # save models and training states
            if current_step % config['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    trainer.save(current_step)
                    trainer.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        trainer.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
