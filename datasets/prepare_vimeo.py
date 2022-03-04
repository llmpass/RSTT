import os
import sys
import shutil
import argparse
from sep_vimeo import sep_vimeo
from generate_LR import generate_LR
from create_lmdb import create_lmdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ProgressBar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Vimeo 90k dataset.')
    parser.add_argument('--path', 
                        help='path to Vimeo dataset', 
                        required=True, 
                        type=str)
    path = parser.parse_args().path
    print(path)

    # Separate datasets
    data_path = os.path.join(path, 'sequences')
    for mode in ['train', 'fast_test', 'medium_test', 'slow_test']:
        print('Separate {} dataset'.format(mode))
        save_path = os.path.join(data_path, mode)
        txt_file = os.path.join(path, 'sep_{}list.txt'.format(mode))
        sep_vimeo(data_path, save_path, txt_file)

    # Genereate LR images
    for mode in ['train', 'fast_test', 'medium_test', 'slow_test']:
        print('Generate low resolution images for {} dataset'.format(mode))
        data_path = os.path.join(path, 'sequences', mode)
        save_path = os.path.join(path, 'sequences_LR')
        
        dirpaths = []
        for dirpath, dirnames, filenames in os.walk(data_path):
            if len(dirnames) == 0:
                dirpaths.append(dirpath)

        print("Generating LR images ...")
        pbar = ProgressBar(len(dirpaths))
        for dirpath in dirpaths:
            pbar.update('Generate {}'.format('/'.join(dirpath.split('/')[-2:])))
            output_path = os.path.join(save_path, '/'.join(dirpath.split('/')[-3:]))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            generate_LR(dirpath, output_path, 4)
    
    # Create lmdb
    lr_data_path = os.path.join(path, 'sequences_LR', 'train')
    hr_data_path = os.path.join(path, 'sequences', 'train')
    lr_save_path = os.path.join(path, 'vimeo_LR.lmdb')
    hr_save_path = os.path.join(path, 'vimeo_HR.lmdb')
    text_path = os.path.join(path, 'sep_trainlist.txt')

    print("Creating lmdb for LR images ...")
    create_lmdb(lr_data_path, text_path, lr_save_path)
    print("Creating lmdb for HR images ...")
    create_lmdb(hr_data_path, text_path, hr_save_path)
    shutil.copyfile(os.path.join(hr_save_path, 'Vimeo_keys.pkl'), os.path.join(path, 'Vimeo_keys.pkl'))