import os
from collections import OrderedDict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def OrderedYaml():
    '''yaml OrderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

Loader, Dumper = OrderedYaml()

def parse_config(config_path, is_train=True):
    with open(config_path, mode='r') as f:
        config = yaml.load(f, Loader=Loader)

    # Export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in config['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    config['is_train'] = is_train
    scale = config['scale']

    # Dataset
    if is_train:
        config['dataset']['scale'] = scale
        is_lmdb = False
        config['dataset']['dataroot_HR'] = os.path.expanduser(config['dataset']['dataroot_HR'])
        if config['dataset']['dataroot_HR'].endswith('lmdb'):
            is_lmdb = True
        config['dataset']['dataroot_LR'] = os.path.expanduser(config['dataset']['dataroot_LR'])
        if config['dataset']['dataroot_LR'].endswith('lmdb'):
            is_lmdb = True
        config['dataset']['data_type'] = 'lmdb' if is_lmdb else 'img'
    else:
        config['dataset']['dataset_root'] = os.path.expanduser(config['dataset']['dataset_root'])

    # Path
    for key, path in config['path'].items():
        if path and key in config['path'] and key != 'strict_load':
            config['path'][key] = os.path.expanduser(path)

    config['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    if is_train:
        experiments_root = os.path.join(config['path']['root'], 'experiments', config['name'])
        config['path']['experiments_root'] = experiments_root
        config['path']['log'] = experiments_root
        config['path']['models'] = os.path.join(experiments_root, 'models')
        config['path']['training_state'] = os.path.join(experiments_root, 'training_state')

        # change some options for debug mode
        if 'debug' in config['name']:
            config['train']['val_freq'] = 8
            config['logger']['print_freq'] = 1
            config['logger']['save_checkpoint_freq'] = 8
    else:  
        if 'vid' in config['dataset']['name'].lower():
            save_path = os.path.join(config['path']['output_dir'], config['dataset']['name'])
        else:
            save_path = os.path.join(config['path']['output_dir'], config['dataset']['name'], config['dataset']['mode'])
        config['path']['save_path'] = save_path

    # network
    config['network']['scale'] = scale

    return config 


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


# convert to NoneDict, which return None for missing key.
class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
