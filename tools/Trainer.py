import logging
from collections import OrderedDict

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models import create_model
from utils import CharbonnierLoss, CosineAnnealingLR_Restart

logger = logging.getLogger('base')


class Trainer():
    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda' if config['gpu_ids'] is not None else 'cpu')
        self.is_train = config['is_train']

        if config['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_config = config['train']

        # define model and load pretrained model
        self.model = create_model(config).to(self.device)

        if config['dist']:
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])
        else:
            self.model = DataParallel(self.model)
        self.get_total_parameters(self.model)

        self.load()

        if self.is_train:
            self.model.train()

            # loss
            self.criterion= CharbonnierLoss().to(self.device)

            # optimizer
            wd = train_config['weight_decay'] if train_config['weight_decay'] else 0
            optim_params = []
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer = torch.optim.AdamW(optim_params, lr=train_config['lr'],
                                                weight_decay=wd,
                                                betas=(train_config['beta1'], train_config['beta2']))
            # schedulers
            self.scheduler = CosineAnnealingLR_Restart(
                    self.optimizer, train_config['T_period'], 
                    eta_min=train_config['eta_min'],
                    restarts=train_config['restarts'], 
                    weights=train_config['restart_weights']
            )

            self.log_dict = OrderedDict()

    def train_one_sample(self, data, step):
        self.inputs = data['LRs'].to(self.device)
        self.targets = data['HRs'].to(self.device)

        self.optimizer.zero_grad()
        self.outputs = self.model(self.inputs)
        loss = self.criterion(self.outputs, self.targets)
        loss.backward()
        self.optimizer.step()

        # set log
        self.log_dict['loss'] = loss.item()

        # update learning rate
        self.update_learning_rate(step, warmup_iter=self.config['train']['warmup_iter'])

    def get_current_log(self):
        return self.log_dict

    def get_total_parameters(self, model):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        total_parameters = sum(map(lambda x: x.numel(), model.parameters()))

        net_struc_str = '{}'.format(model.__class__.__name__)

        if self.rank <= 0:
            logger.info('Model structure: {}, with parameters: {:,d}'.format(net_struc_str, total_parameters))

    def load(self):
        load_path = self.config['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model [{:s}] ...'.format(load_path))
            self.load_model(load_path, self.model, self.config['path']['strict_load'])

    def save(self, iter_label):
        self.save_model(self.model, iter_label)

    def _set_lr(self, lr_groups):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups = [v['initial_lr'] for v in self.optimizer.param_groups]
        return init_lr_groups

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        self.scheduler.step()
        # set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_groups = self._get_init_lr()
            # modify warming-up learning rates
            warmup_lr = [v / warmup_iter * cur_iter for v in init_lr_groups]
            # set learning rate
            self._set_lr(warmup_lr)

    def get_current_learning_rate(self):
        lr_l = []
        for param_group in self.optimizer.param_groups:
            lr_l.append(param_group['lr'])
        return lr_l


    def save_model(self, model, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(self.config['path']['models'], save_filename)
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_model(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {
            'epoch': epoch, 
            'iter': iter_step, 
            'scheduler': self.scheduler.state_dict(), 
            'optimizer': self.optimizer.state_dict()
        }
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.config['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizer = resume_state['optimizer']
        resume_scheduler = resume_state['scheduler']
        self.optimizer.load_state_dict(resume_optimizer)
        self.scheduler.load_state_dict(resume_scheduler)
