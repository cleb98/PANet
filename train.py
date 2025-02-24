"""Improved Training Script with Better Logging"""
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from tqdm import tqdm  # Progress bar

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from util.visual_utils import decode_and_apply_mask_overlay
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')

    labels = CLASS_LABELS[data_name][_config['label_sets']]
    print('classes: ', labels)

    transforms = Compose([Resize(size=_config['input_size']), RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    _log.info('###### Training Start ######')
    log_loss = {'loss': 0, 'align_loss': 0}
    total_batches = len(trainloader)

    start_time = time.time()

    for i_iter, sample_batched in enumerate(tqdm(trainloader, desc="Training Progress", total=total_batches)):
        batch_start_time = time.time()

        # Prepare input
        support_images = [[shot.cuda() for shot in way] for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
        query_loss = criterion(query_pred, query_labels) #

        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss


        # Monitor training time and memory usage
        batch_time = time.time() - batch_start_time
        gpu_mem = torch.cuda.memory_allocated(_config['gpu_id']) / (1024 ** 2)  # MB

        # Print logs at interval
        if (i_iter + 1) % _config['print_interval'] == 0:
            avg_loss = log_loss['loss'] / (i_iter + 1)
            avg_align_loss = log_loss['align_loss'] / (i_iter + 1)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (i_iter + 1)) * total_batches
            remaining_time = estimated_total_time - elapsed_time

            print(f"[Iter {i_iter+1}/{total_batches}] "
                  f"Loss: {avg_loss:.4f}, Align Loss: {avg_align_loss:.4f} "
                  f"| Time per batch: {batch_time:.2f}s "
                  f"| GPU Mem: {gpu_mem:.2f} MB "
                  f"| Remaining time: {remaining_time/60:.2f} min")
        # Save model periodically
        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
    _log.info(f"Training Completed in {(time.time() - start_time) / 60:.2f} minutes")
