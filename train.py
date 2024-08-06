import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import json

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.transbts.transbts_downsample8x import get_default as TransBTS
from models.unet.unet3d import get_default as UNet3D
from models.vit.vit2d import VisionTransformer2D as ViT2D
from models.unet.unet2d import get_default as UNet2D

import criterion
from data.dataset import BraTS3D, BraTS2D
from torch.utils.data import DataLoader
from utils import all_reduce_tensor, log_args, adjust_learning_rate, Accumulator, setup, cleanup
from torch import nn
from config import Config, PediatricConfig, AdultConfig
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


local_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
date = local_time.split(' ')[0]

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='transbts', type=str)

parser.add_argument('--dataset', default='brats_ped_2023', type=str)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=6, type=int)

parser.add_argument('--start_epoch', default=1, type=int)

parser.add_argument('--end_epoch', default=400, type=int)

parser.add_argument('--save_freq', default=5000, type=int)

parser.add_argument('--resume', default=f'kidnet-brats_ped_2023-{date}', type=str)

parser.add_argument('--load', default=False, type=bool)

args = parser.parse_args()

if args.dataset == "brats_ped_2023":
    config = PediatricConfig()
elif args.dataset == "brats_2019":
    config = AdultConfig()

root = config.BRATS_DIR
train_dir = config.BRATS_TRAIN["dir"]
train_list = config.BRATS_TRAIN["list"]


def main():
    local_rank, world_size = setup(args.seed)
    if local_rank == 0:
        log_dir = os.path.join(config.LOG_DIR, args.model + "-" + args.dataset + "-" + date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('------------------------------------Running Information----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('--------------------------------------Model Training-------------------------------------')

    tra_st = (os.path.join(train_dir, train_list), train_dir, "train")
    val_st = (os.path.join(train_dir, train_list), train_dir, "valid")
    if args.model == "unet3d":
        model = UNet3D()
        train_set = BraTS3D(*tra_st)
        valid_set = BraTS3D(*val_st)
    elif args.model == "transbts":
        model = TransBTS()
        train_set = BraTS3D(*tra_st)
        valid_set = BraTS3D(*val_st)
    elif args.model == "vit2d":
        model = ViT2D()
        train_set = BraTS2D(*tra_st)
        valid_set = BraTS2D(*val_st)
    elif args.model == "unet2d":
        model = UNet2D()
        train_set = BraTS2D(*tra_st)
        valid_set = BraTS2D(*val_st)

    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                                                find_unused_parameters=True)
    model.train()

    scaler = GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    total_params = sum(p.numel() for p in model.parameters())

    logging.info('Total number of parameters: {}'.format(total_params))
    
    crit = getattr(criterion, args.criterion)

    if local_rank == 0:
        checkpoint_dir = os.path.join(config.CHECK_POINT_DIR, args.model + "-" + args.dataset + "-" + date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    resume_path = os.path.join(config.CHECK_POINT_DIR, args.resume)

    if os.path.isfile(resume_path) and args.load:
        logging.info('Loading checkpoint: {}'.format(args.resume))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('Train the model from scratch')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
    logging.info('The number of samples for training: {}'.format(len(train_set)))
    logging.info('The number of samples for validation: {}'.format(len(valid_set)))

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // world_size,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    
    valid_loader = DataLoader(dataset=valid_set, sampler=valid_sampler, batch_size=args.batch_size // world_size,
                              drop_last=False, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    stats_train = []
    stats_valid = []

    init_dice = validate(model, valid_loader, crit, local_rank, active=world_size)
    logging.info('Epoch: 0 -- Initial Stage -- softmax dice loss: {:.5f} | dice score for class 1: {:.4f} | dice score for class 2: {:.4f} | dice score for class 3: {:.4f}'
                             .format(*init_dice.avg()))

    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        setproctitle.setproctitle('{}: {}/{}'.format(args.model, epoch, args.end_epoch))
        start_epoch_time = time.time()

        metric_train = Accumulator(4)

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            with autocast():
                output = model(x)
                loss, score1, score2, score3 = crit(output, target)

            optimizer.zero_grad()
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            reduce_loss = all_reduce_tensor(loss, active=world_size).data.cpu().numpy()
            reduce_score1 = all_reduce_tensor(score1, active=world_size).data.cpu().numpy()
            reduce_score2 = all_reduce_tensor(score2, active=world_size).data.cpu().numpy()
            reduce_score3 = all_reduce_tensor(score3, active=world_size).data.cpu().numpy()

            metric_train.add(reduce_loss, reduce_score1, reduce_score2, reduce_score3)

            if local_rank == 0:
                logging.info('Epoch: {}, Iter: {} -- loss: {:.5f} | 1: {:.4f} | 2: {:.4f} | 3: {:.4f}'
                             .format(epoch, i, reduce_loss, reduce_score1, reduce_score2, reduce_score3))
        
        metric_valid = validate(model, valid_loader, crit, local_rank, active=world_size)
        end_epoch_time = time.time()
        if local_rank == 0:
            logging.info('Epoch: {} -- Training Stage -- softmax dice loss: {:.5f} | dice score for class 1: {:.4f} | dice score for class 2: {:.4f} | dice score for class 3: {:.4f}'
                             .format(epoch, *metric_train.avg()))
            
            logging.info('Epoch: {} -- Validation Stage -- softmax dice loss: {:.5f} | dice score for class 1: {:.4f} | dice score for class 2: {:.4f} | dice score for class 3: {:.4f}'
                             .format(epoch, *metric_valid.avg()))
            
            stats_train.append(metric_train.data)
            stats_valid.append(metric_valid.data)

            if ((epoch) % int(args.save_freq) == 0 and epoch != args.end_epoch) or args.end_epoch - epoch == 1 or args.end_epoch - epoch == 2 or args.end_epoch - epoch == 3:
                file_name = os.path.join(checkpoint_dir, '{}-{}-epoch_{}.pth'.format(args.model, args.dataset, epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                }, file_name)
        if local_rank == 0:
            epoch_time_minute = (end_epoch_time - start_epoch_time) / 60
            remaining_time_hour = (args.end_epoch - epoch) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes.'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours.'.format(remaining_time_hour))

    if local_rank == 0:
        if (not os.path.exists(config.COLLECTION_DIR)):
            os.makedirs(config.COLLECTION_DIR, exist_ok=True)
        with open(os.path.join(config.COLLECTION_DIR, f"{args.model}-{args.dataset}-train.json"), "w") as f:
            json.dump(stats_train, f)

        with open(os.path.join(config.COLLECTION_DIR, f"{args.model}-{args.dataset}-valid.json"), "w") as f:
            json.dump(stats_valid, f)

        final_name = os.path.join(checkpoint_dir, '{}-{}-last.pth'.format(args.model, args.dataset))

        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    cleanup()
    logging.info('-----------------------------------Training Process Over---------------------------------')


def validate(model, loader, criterion, local_rank, active):
    model.eval()
    metric = Accumulator(4)
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            x, target = data
            x = x.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            with autocast():
                output = model(x)
                loss, score1, score2, score3 = criterion(output, target)
            reduce_loss = all_reduce_tensor(loss, active=active).data.cpu().numpy()
            sum_score1 = all_reduce_tensor(score1, active=active).data.cpu().numpy()
            sum_score2 = all_reduce_tensor(score2, active=active).data.cpu().numpy()
            sum_score3 = all_reduce_tensor(score3, active=active).data.cpu().numpy()
            
            metric.add(reduce_loss, sum_score1, sum_score2, sum_score3)
    model.train()
    return metric


if __name__ == '__main__':
    print(f"Number of devices available: {torch.cuda.device_count()}")
    cudnn.enabled = True
    cudnn.benchmark = True
    main()