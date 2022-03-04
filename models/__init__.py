import logging
from .RSTT import RSTT
logger = logging.getLogger('base')


def create_model(config):
    network_config = config['network']

    if config['model'] == 'RSTT':
        model = RSTT(
            embed_dim=network_config['embed_dim'],
            depths=network_config['depths'],
            num_heads=network_config['num_heads'],
            window_sizes=network_config['window_sizes'],
            back_RBs=network_config['back_RBs']
        ) 
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(config['model']))

    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))

    return model