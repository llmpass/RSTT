import os
import logging

def check_resume(config, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if config['path']['resume_state']:
        if config['path'].get('pretrain_model', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        config['path']['pretrain_model'] = os.path.join(config['path']['models'],
                                                   '{}.pth'.format(resume_iter))
        logger.info('Set [pretrain_model] to ' + config['path']['pretrain_model'])
