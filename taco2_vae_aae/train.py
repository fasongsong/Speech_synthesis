import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from torch import nn
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from fp16_optimizer import FP16_Optimizer

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, AAE_D_loss, AAE_G_loss
from logger import Tacotron2Logger
from hparams import create_hparams

import gc


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams, warp_set="og")
    trainset_aug_time = TextMelLoader(hparams.training_files, hparams, warp_set="time")
    trainset_aug_freq = TextMelLoader(hparams.training_files, hparams, warp_set="freq")

    valset = TextMelLoader(hparams.validation_files, hparams, warp_set="og")
    valset_aug_time = TextMelLoader(hparams.validation_files, hparams, warp_set="time")
    valset_aug_freq = TextMelLoader(hparams.validation_files, hparams, warp_set="freq")

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_aug_set = torch.utils.data.ConcatDataset([trainset, trainset_aug_time, trainset_aug_freq])
    train_sampler = DistributedSampler(train_aug_set) if hparams.distributed_run else None

    if hparams.distributed_run:
        train_loader = DataLoader(train_aug_set,
                                  num_workers=35, shuffle=False,
                                  sampler=train_sampler,
                                  batch_size=hparams.batch_size, pin_memory=False,
                                  drop_last=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(torch.utils.data.ConcatDataset([trainset, trainset_aug_time, trainset_aug_freq]), num_workers=35, shuffle=True,
                                  sampler=None,
                                  batch_size=hparams.batch_size, pin_memory=False,
                                  drop_last=True, collate_fn=collate_fn)

    del trainset, trainset_aug_time, trainset_aug_freq

    return train_loader, valset, valset_aug_time, valset_aug_freq, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model = batchnorm_to_float(model.half())
        model.decoder.attention_layer.score_mask_value = float(finfo('float16').min)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, ae_optimizer, d_optimizer, g_optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    ae_optimizer.load_state_dict(checkpoint_dict['ae_optimizer'])
    d_optimizer.load_state_dict(checkpoint_dict['d_optimizer'])
    g_optimizer.load_state_dict(checkpoint_dict['g_optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, ae_optimizer, d_optimizer, g_optimizer, learning_rate, iteration


def save_checkpoint(model, ae_optimizer, d_optimizer, g_optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'ae_optimizer': ae_optimizer.state_dict(), 'd_optimizer': d_optimizer.state_dict(), 'g_optimizer': g_optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, ae_criterion, d_criterion, g_criterion, valset, valset_aug_time, valset_aug_freq, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_aug_set = torch.utils.data.ConcatDataset([valset, valset_aug_time, valset_aug_freq])
        val_sampler = DistributedSampler(val_aug_set) if distributed_run else None

        if distributed_run:
            val_loader = DataLoader(val_aug_set, sampler=val_sampler, num_workers=35,
                                    shuffle=False, batch_size=batch_size,
                                    pin_memory=False, drop_last=True, collate_fn=collate_fn)
        else:
            val_loader = DataLoader(torch.utils.data.ConcatDataset([valset, valset_aug_time, valset_aug_freq]), sampler=None, num_workers=35,
                                    shuffle=True, batch_size=batch_size,
                                    pin_memory=False, drop_last=True, collate_fn=collate_fn)

        del valset, valset_aug_time, valset_aug_freq

        val_loss = 0.0
        sp_pos = 0
        sp_neg = 0
        au_pos = 0
        au_neg = 0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            ae_loss, recon_loss, speaker_loss, augment_loss, alignment_loss = ae_criterion(y_pred, y, iteration)
            d_loss = d_criterion(y_pred)
            g_loss = g_criterion(y_pred)

            #########################speaker, augmentation classifier accuracy #######################################
            _,_, _, _, speaker_out, aug_out, _, _ = y_pred
            speaker_target, aug_target = y[2], y[3]
            for e in range(hparams.batch_size):
                sp_t = torch.argmax(speaker_target[e])
                sp_p = torch.argmax(speaker_out[e])
                au_t = torch.argmax(aug_target[e])
                au_p = torch.argmax(aug_out[e])
                if sp_t == sp_p:
                    sp_pos += 1
                else:
                    sp_neg += 1
                if au_t == au_p:
                    au_pos += 1
                else:
                    au_neg += 1
            #############################################################################################################
            if distributed_run:
                reduced_val_loss = reduce_tensor(ae_loss.data, n_gpus).item()
                reduced_val_loss += reduce_tensor(d_loss.data, n_gpus).item()
                reduced_val_loss += reduce_tensor(g_loss.data, n_gpus).item()
            else:
                reduced_val_loss = ae_loss.item()
                reduced_val_loss += d_loss.item()
                reduced_val_loss += g_loss.item()

            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)
        speaker_acc = float(sp_pos) / float(sp_pos + sp_neg)
        augment_acc = float(au_pos) / float(au_pos + au_neg)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration, speaker_acc, augment_acc)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout
    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    torch.nn.functional.sigmoid
    model = load_model(hparams)

    learning_rate = hparams.learning_rate
    #lr = args.lr * (0.1 ** (epoch // 30))
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=hparams.weight_decay)
    d_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=hparams.weight_decay)
    g_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=hparams.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, dampening=0, weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        optimizer = FP16_Optimizer(
            optimizer, dynamic_loss_scale=hparams.dynamic_loss_scaling)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    ae_criterion = Tacotron2Loss(hparams)
    d_criterion = AAE_D_loss()
    g_criterion = AAE_G_loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, _, _, _, _ = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model)
        else:
            model, ae_optimizer, d_optimizer, g_optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, ae_optimizer, d_optimizer, g_optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1

            epoch_offset = max(0, int(iteration / len(train_loader)))


    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    step = 0
    for epoch in range(epoch_offset, hparams.epochs):
        train_loader, valset, valset_aug_time, valset_aug_freq, collate_fn = prepare_dataloaders(hparams)
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group1 in ae_optimizer.param_groups:          ################## 체크하기.
                param_group1['lr'] = learning_rate
            for param_group2 in d_optimizer.param_groups:          ################## 체크하기.
                param_group2['lr'] = learning_rate/5
            for param_group3 in g_optimizer.param_groups:          ################## 체크하기.
                param_group3['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            ae_loss, recon_loss, speaker_loss, augment_loss, alignment_loss = ae_criterion(y_pred, y, iteration)
            d_loss = d_criterion(y_pred)
            g_loss = g_criterion(y_pred)

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(ae_loss.data, n_gpus).item()
                reduced_loss += reduce_tensor(d_loss.data, n_gpus).item()
                reduced_loss += reduce_tensor(g_loss.data, n_gpus).item()
            else:
                reduced_loss = ae_loss.item()
                reduced_loss += d_loss.item()
                reduced_loss += g_loss.item()

            if hparams.fp16_run:
                optimizer.backward(loss)
                grad_norm = optimizer.clip_fp32_grads(hparams.grad_clip_thresh)
            else:
                ae_loss.backward(retain_graph=True)
                d_loss.backward(retain_graph=True)
                g_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            ae_optimizer.step()
            d_optimizer.step()
            g_optimizer.step()

            overflow = ae_optimizer.overflow if hparams.fp16_run else False

            if not overflow and not math.isnan(reduced_loss) and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(reduced_loss, grad_norm, learning_rate, duration, recon_loss, \
                                    speaker_loss, augment_loss, alignment_loss, d_loss, g_loss, iteration)

            if not overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, ae_criterion, d_criterion, g_criterion, valset, valset_aug_time, valset_aug_freq, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, ae_optimizer, d_optimizer, g_optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1
            # print('Memory Usage:')
            # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            # print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
            del ae_loss, d_loss, g_loss, recon_loss, speaker_loss, augment_loss, alignment_loss
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', default='./check_aae', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', default='./logs', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Time warping: ", hparams.mel_time_warping)
    print("Freq warping: ", hparams.mel_freq_warping)
    print("Batch_size: ", hparams.batch_size)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
    # train("./check_point", "./logs", None,
    #       args.warm_start, 4, args.rank, args.group_name, hparams)
