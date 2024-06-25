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
from network.transbts import TransBTS
from network.model.unet3d import UNet3D
import torch.distributed as dist

import criterion
from prepare.data import BraTS
from torch.utils.data import DataLoader
from utils import all_reduce_tensor, log_args, adjust_learning_rate, Accumulator
# from tensorboardX import SummaryWriter
from torch import nn
from config import Config, PediatricConfig, AdultConfig


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
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

parser.add_argument('--gpu', default='0,1,2', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=6, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=400, type=int)

parser.add_argument('--save_freq', default=5000, type=int)

parser.add_argument('--resume', default=f'kidnet-brats_ped_2023-{date}', type=str)

parser.add_argument('--load', default=False, type=bool)

args = parser.parse_args()

local_rank = int(os.environ['LOCAL_RANK'])

if args.dataset == "brats_ped_2023":
    config = PediatricConfig()
elif args.dataset == "brats_2019":
    config = AdultConfig()

root = config.BRATS_DIR
train_dir = config.BRATS_TRAIN["dir"]
train_list = config.BRATS_TRAIN["list"]


def main_worker():
    if local_rank == 0:
        log_dir = os.path.join(config.LOG_DIR, args.model + "-" + args.dataset + "-" + date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('------------------------------------Running Information----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('--------------------------------------Model Training-------------------------------------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(local_rank)

    if args.model == "unet3d":
        model = UNet3D(4, 4)
        find_unused_parameters = False
    elif args.model == "transbts":
        model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        find_unused_parameters = True
    else:
        raise ValueError(f"Invalid model {args.model}, check first.")

    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                find_unused_parameters=find_unused_parameters)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    total_params = sum(p.numel() for p in model.parameters())

    logging.info('Total number of parameters: {}'.format(total_params))
    

    crit = getattr(criterion, args.criterion)

    if local_rank == 0:
        checkpoint_dir = os.path.join(config.CHECK_POINT_DIR, args.model + "-" + args.dataset + "-" + date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # writer = SummaryWriter()

    if os.path.isfile(args.resume) and args.load:
        logging.info('Loading checkpoint: {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('Train the model from scratch')

    train_set = BraTS(os.path.join(train_dir, train_list), train_dir, "train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('The number of samples for training: {}'.format(len(train_set)))


    num_gpu = len(args.gpu.split(","))

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    stats = []

    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)
        setproctitle.setproctitle('{}: {}/{}'.format(args.model, epoch+1, args.end_epoch))
        start_epoch_time = time.time()

        metric = Accumulator(4)

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            output = model(x)

            loss, score1, score2, score3 = crit(output, target)
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            reduce_score1 = all_reduce_tensor(score1, world_size=num_gpu).data.cpu().numpy()
            reduce_score2 = all_reduce_tensor(score2, world_size=num_gpu).data.cpu().numpy()
            reduce_score3 = all_reduce_tensor(score3, world_size=num_gpu).data.cpu().numpy()

            metric.add(reduce_loss, reduce_score1, reduce_score2, reduce_score3)

            if local_rank == 0:
                logging.info('Epoch: {}, Iter: {} -- loss: {:.5f} | 1: {:.4f} | 2: {:.4f} | 3: {:.4f}'
                             .format(epoch, i, reduce_loss, reduce_score1, reduce_score2, reduce_score3))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_epoch_time = time.time()
        if local_rank == 0:
            logging.info('Epoch: {} -- softmax dice loss: {:.5f} | dice score for class 1: {:.4f} | dice score for class 2: {:.4f} | dice score for class 3: {:.4f}'
                             .format(epoch, *metric.avg()))
            
            stats.append(metric.data)

            if (epoch + 1) % int(args.save_freq) == 0:
                file_name = os.path.join(checkpoint_dir, '{}-{}-epoch_{}.pth'.format(args.model, args.dataset, epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                }, file_name)

            # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('loss', reduce_loss, epoch)
            # writer.add_scalar('score1', reduce_score1, epoch)
            # writer.add_scalar('score2', reduce_score2, epoch)
            # writer.add_scalar('score3', reduce_score3, epoch)

        if local_rank == 0:
            epoch_time_minute = (end_epoch_time - start_epoch_time) / 60
            remaining_time_hour = (args.end_epoch - (epoch + 1)) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes.'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours.'.format(remaining_time_hour))

    if local_rank == 0:
        # writer.close()

        with open(f"./{args.model}-{args.dataset}.json", "w") as f:
            json.dump(stats, f)

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

    logging.info('-----------------------------------Training Process Over---------------------------------')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Number of devices available: {torch.cuda.device_count()}")
    print(f"Process {os.getpid()} is using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    assert torch.cuda.is_available(), "Only CUDA version is supported."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()