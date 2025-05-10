#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import json
import copy
from tqdm import tqdm

from loader import TwoCropsTransform
from datautils import TwoCropsTransform, AddWhiteNoise, VolumeChange, AddFade, WaveTimeStretch, PitchShift, CodecApply, AddEnvironmentalNoise, ResampleAugmentation, AddEchoes, TimeShift, AddZeroPadding, genSpoof_list, Dataset_ASVspoof2019_train, pad_or_clip_batch
from model import MoCo_v2
from aasist import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool, CONV, Residual_block, AasistEncoder

import torch
import torchaudio
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#-----------------------------------------------------------------------------------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Configurations
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)

with open("/home/hwang-gyuhan/Workspace/ND/config.conf", "r") as f_json:
    config = json.load(f_json)

def load_model(config: dict):
    aasist_config_path = config['aasist_config_path']
    with open(aasist_config_path, "r") as f_json:
        aasist_config = json.load(f_json)
    
    return aasist_config["model_config"]

d_args = load_model(config)
encoder = AasistEncoder(d_args=d_args)

model_names = ["aasist_encoder"]
parser = argparse.ArgumentParser(description="PyTorch Test Training")
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=150, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=24,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0005,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data manipulations"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is None:
        args.gpu = 0  
    else:
        main_worker(args, args.gpu)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args) -> None:
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

# data loading code
database_path = config["database_path"]
cut_length = 64600
batch_size = 24
score_save_path = "./results/scores.txt"
augmentations_on_cpu = None
manipulations = None

d_label_trn, file_eval, utt2spk = genSpoof_list(
    dir_meta=os.path.join(database_path, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
    is_train=False,
    is_eval=False
)
print('no. of ASVspoof 2019 LA evaluating trials', len(file_eval))

asvspoof_LA_train_dataset = Dataset_ASVspoof2019_train(
    list_IDs=file_eval,
    labels=d_label_trn,
    base_dir=os.path.join(database_path, 'ASVspoof2019_LA_train/'),
    cut_length=cut_length,
    utt2spk=utt2spk
)

asvspoof_2019_LA_train_dataloader = DataLoader(
    asvspoof_LA_train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    pin_memory=True
)
# Main part where you create dataset with manipulations and TwoCropsTransform
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
noise_dataset_path = config["noise_dataset_path"]
manipulations = {
    "no_augmentation": None,
    "volume_change_50": torchaudio.transforms.Vol(gain=0.5,gain_type='amplitude'),
    "volume_change_10": torchaudio.transforms.Vol(gain=0.1,gain_type='amplitude'),
    "white_noise_15": AddWhiteNoise(max_snr_db = 15, min_snr_db=15),
    "white_noise_20": AddWhiteNoise(max_snr_db = 20, min_snr_db=20),
    "white_noise_25": AddWhiteNoise(max_snr_db = 25, min_snr_db=25),
    "env_noise_wind": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="wind", noise_dataset_path=noise_dataset_path),
    "env_noise_footsteps": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="footsteps", noise_dataset_path=noise_dataset_path),
    "env_noise_breathing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="breathing", noise_dataset_path=noise_dataset_path),
    "env_noise_coughing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="coughing", noise_dataset_path=noise_dataset_path),
    "env_noise_rain": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="rain", noise_dataset_path=noise_dataset_path),
    "env_noise_clock_tick": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="clock_tick", noise_dataset_path=noise_dataset_path),
    "env_noise_sneezing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="sneezing", noise_dataset_path=noise_dataset_path),
    "pitchshift_up_110": PitchShift(max_pitch=1.10, min_pitch=1.10, bins_per_octave=12),
    "pitchshift_up_105": PitchShift(max_pitch=1.05, min_pitch=1.05, bins_per_octave=12),
    "pitchshift_down_095": PitchShift(max_pitch=0.95, min_pitch=0.95, bins_per_octave=12),
    "pitchshift_down_090": PitchShift(max_pitch=0.90, min_pitch=0.90, bins_per_octave=12),
    "timestretch_110": WaveTimeStretch(max_ratio=1.10, min_ratio=1.10, n_fft=128),
    "timestretch_105": WaveTimeStretch(max_ratio=1.05, min_ratio=1.05, n_fft=128),
    "timestretch_095": WaveTimeStretch(max_ratio=0.95, min_ratio=0.95, n_fft=128),
    "timestretch_090": WaveTimeStretch(max_ratio=0.90, min_ratio=0.90, n_fft=128),
    "echoes_1000_02": AddEchoes(max_delay=1000, max_strengh=0.2, min_delay=1000, min_strength=0.2),
    "echoes_1000_05": AddEchoes(max_delay=1000, max_strengh=0.5, min_delay=1000, min_strength=0.5),
    "echoes_2000_05": AddEchoes(max_delay=2000, max_strengh=0.5, min_delay=2000, min_strength=0.5),
    "time_shift_1600": TimeShift(max_shift=1600, min_shift=1600),
    "time_shift_16000": TimeShift(max_shift=16000, min_shift=16000),
    "time_shift_32000": TimeShift(max_shift=32000, min_shift=32000),
    "fade_50_linear": AddFade(max_fade_size=0.5,fade_shape='linear', fix_fade_size=True),
    "fade_30_linear": AddFade(max_fade_size=0.3,fade_shape='linear', fix_fade_size=True),
    "fade_10_linear": AddFade(max_fade_size=0.1,fade_shape='linear', fix_fade_size=True),
    "fade_50_exponential": AddFade(max_fade_size=0.5,fade_shape='exponential', fix_fade_size=True),
    "fade_50_quarter_sine": AddFade(max_fade_size=0.5,fade_shape='quarter_sine', fix_fade_size=True),
    "fade_50_half_sine": AddFade(max_fade_size=0.5,fade_shape='half_sine', fix_fade_size=True),
    "fade_50_logarithmic": AddFade(max_fade_size=0.5,fade_shape='logarithmic', fix_fade_size=True),
    "resample_15000": ResampleAugmentation([15000], device="cuda"),
    "resample_15500": ResampleAugmentation([15500], device="cuda"),
    "resample_16500": ResampleAugmentation([16500], device="cuda"),
    "resample_17000": ResampleAugmentation([17000], device="cuda"), # device="cuda"
}

class ComposeWithNone:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            if t is not None:
                x = t(x)
        return x
    
with open(score_save_path, 'w') as file:
    pass

for batch_idx, (audio_input, spks, labels) in enumerate(tqdm(asvspoof_2019_LA_train_dataloader)):
    score_list = []  
    audio_input = audio_input.squeeze(1)

    if augmentations_on_cpu is not None:
        audio_input = augmentations_on_cpu(audio_input)

    audio_input = audio_input.to(device)

    for manipulation_name, manipulation in manipulations.items():
        if manipulation is not None:
            audio_input = manipulation(audio_input.to(device))
            
    if audio_input.shape[-1] < cut_length:
        audio_input = audio_input.repeat(1, int(cut_length / audio_input.shape[-1]) + 1)[:, :cut_length]
    elif audio_input.shape[-1] > cut_length:
        audio_input = audio_input[:, :cut_length]
    audio_input = audio_input.to(device)
    print("audio_length", audio_input.shape)

    base_transform = ComposeWithNone(list(manipulations.values()))
    two_crop_transform = TwoCropsTransform(base_transform=base_transform)
    q, k = two_crop_transform(audio_input)

    # Define MoCo_v2 model (assuming 'MoCo_v2' is already implemented)
    model = MoCo_v2(
        encoder_q=encoder,
        encoder_k=encoder,
        queue_feature_dim=encoder.last_hidden
    ).to(device)
    print(model)
    
    q = q.to(device)
    k = k.to(device)
    q_emb = model.encoder_q(q)
    k_emb = model.encoder_k(k)
    
    batch_out = model(audio_input)
    batch_out = batch_out[1]  

    batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel()
    label_list = ['bonafide' if i == 1 else 'spoof' for i in labels]
    score_list.extend(batch_score.tolist())

    with open(score_save_path, 'a+') as fh:
        for label, cm_score in zip(label_list, score_list):
            fh.write('- - {} {}\n'.format(label, cm_score))

print('Scores saved to {}'.format(score_save_path))
get_eval_metrics(score_save_path=score_save_path, plot_figure=False)

   
torch.cuda.set_device(args.gpu)
print(f"Use GPU: {args.gpu} for training")
model.cuda(args.gpu)  

criterion = nn.CrossEntropyLoss().cuda(args.gpu)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

x1, x2 = create_train_dataset_with_two_crops(database_path, batch_size=1024)
out_q, out_k = model(x1.to(device), x2.to(device))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    num_workers=args.workers,
    pin_memory=True,
    sampler=train_sampler,
    drop_last=True,
)

for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename="checkpoint_{:04d}.pth.tar".format(epoch),
        )
 
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()