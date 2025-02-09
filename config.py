"""Experiment Configuration"""
import os
import re
import glob
import itertools
from path import Path

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    ################ SET TRAINING PARAMETERS or TEST PARAMETERS ################
    input_size = (417, 417)
    seed = 1234
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 0
    mode = 'train' #'train' or 'test'
    dataset = 'COCO'  # 'VOC' or 'COCO'
    snapshot = './runs/PANet_COCO_align_sets_0_1way_5shot_[train]/9/snapshots/30000.pth'
    n_ways = 1
    n_shots = 5
    n_queries = 1
    log_tensorboard = f'./runs/{mode}_{dataset}_{n_ways}way_{n_shots}shot_{n_queries}query'

#____________________________________________________________________________________________#

    if mode == 'train':
        dataset = dataset
        n_steps = 30000
        label_sets = 0
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 10000

        model = {
            'align': True,
        }

        task = {
            'n_ways': n_ways,
            'n_shots': n_shots,
            'n_queries': n_queries,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        notrain = False
        snapshot = snapshot
        n_runs = 5
        n_steps = 1000
        batch_size = 1
        scribble_dilation = 0
        bbox = False
        scribble = False

        # Set dataset config from the snapshot string
        if 'VOC' in snapshot:
            dataset = 'VOC'
        elif 'COCO' in snapshot:
            dataset = 'COCO'
        else:
            raise ValueError('Wrong snapshot name !')

        # Set model config from the snapshot string
        model = {}
        for key in ['align',]:
            model[key] = key in snapshot

        # Set label_sets from the snapshot string
        label_sets = int(snapshot.split('_sets_')[1][0])

        # Set task config from the snapshot string
        task = {
            'n_ways': int(re.search("[0-9]+way", snapshot).group(0)[:-3]),
            'n_shots': int(re.search("[0-9]+shot", snapshot).group(0)[:-4]),
            'n_queries': 1,
        }

    else:
        raise ValueError('Wrong configuration for "mode" !')

    print(f"Training mode: {mode}, Dataset: {dataset}")


    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]'])


    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/vgg16-397923af.pth',
        'VOC':{'data_dir': '../../data/Pascal/VOCdevkit/VOC2012/',
               'data_split': 'trainaug',},

        # 'COCO':{'data_dir': '../../data/COCO/',
        #         'data_split': 'train',},
        'COCO':{'data_dir': '/work/tesi_cbellucci/coco',
                'data_split': 'train',}
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    if config['mode'] == 'test':
        if config['notrain']:
            exp_name += '_notrain'
        if config['scribble']:
            exp_name += '_scribble'
        if config['bbox']:
            exp_name += '_bbox'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
