'''Create dataset and dataloader'''
import logging
import torch
import torch.utils.data
from .Vimeo import VimeoDataset


def create_dataloader(dataset, dataset_config, config, sampler):
    if config['dist']:
        world_size = torch.distributed.get_world_size()
        num_workers = dataset_config['n_workers']
        assert dataset_config['batch_size'] % world_size == 0
        batch_size = dataset_config['batch_size'] // world_size
        shuffle = False
    else:
        num_workers = dataset_config['n_workers'] * len(config['gpu_ids'])
        batch_size = dataset_config['batch_size']
        shuffle = True
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)

def create_dataset(dataset_config):
    if dataset_config['name'] == 'Vimeo90k_septuplet':
        dataset = VimeoDataset(dataset_config)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(dataset_config['name']))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_config['name']))
    return dataset
