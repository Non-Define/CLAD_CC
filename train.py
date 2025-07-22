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
import gc
from tqdm import tqdm

from datautils import AddWhiteNoise, VolumeChange, AddFade, WaveTimeStretch, PitchShift, CodecApply, AddEnvironmentalNoise, ResampleAugmentation, AddEchoes, TimeShift, FreqMask, AddZeroPadding, genSpoof_train_list, Dataset_ASVspoof2019, pad_or_clip_batch
from model import ConvLayers, SELayer, SERe2blocks, BiLSTM, BLDL, GraphAttentionLayer, GraphPool, STJGAT, Permute, Model
from transformers import Wav2Vec2Model

import torch
import torchaudio
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#-----------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Configurations
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)

with open("/home/cnrl/Workspace/ND/config.conf", "r") as f_json:
    config = json.load(f_json)

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
    "--arch", default="dual branch", help="model architecture"
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
    default=12,
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
parser.add_argument('--cos', action='store_true', help='Use cosine annealing learning rate')
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
# Distributed Learning
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

def main() -> None:
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

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    args.distributed = False
    args.multiprocessing_distributed = False

    ngpus_per_node = 1 
    main_worker(args.gpu, ngpus_per_node, args)

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
        
    database_path = config["database_path"]
    cut_length = 64600
    batch_size = 12
    score_save_path = "./results/scores.txt"
    augmentations_on_cpu = None
    augmentations = None
    
    # Define XLSR, SE-Re2blocks + STJ-GAT + BLDL
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    encoder = Model().to(device)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(encoder.parameters()),
        args.lr,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            encoder.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # data loading code
    d_label_trn, file_train, utt2spk = genSpoof_train_list(
        dir_meta=os.path.join(database_path, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
        is_train=True,
        is_eval=False
    )
    print('no. of ASVspoof 2019 LA training trials', len(file_train))

    asvspoof_LA_train_dataset = Dataset_ASVspoof2019(
        list_IDs=file_train,
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
    
    total_params_model = sum(p.numel() for p in model.parameters())
    total_params_encoder = sum(p.numel() for p in encoder.parameters())
    print(f"Wav2Vec2: {total_params_model:,} parameters | Encoder: {total_params_encoder:,} parameters | Total: {total_params_model + total_params_encoder:,} parameters")
    
    env_noise_dataset_path = config["env_noise_dataset_path"]
    manipulations = {
        "no_augmentation": None,
        "volume_change_50": torchaudio.transforms.Vol(gain=0.5,gain_type='amplitude'),
        "volume_change_10": torchaudio.transforms.Vol(gain=0.1,gain_type='amplitude'),
        "white_noise_15": AddWhiteNoise(max_snr_db = 15, min_snr_db=15),
        "white_noise_20": AddWhiteNoise(max_snr_db = 20, min_snr_db=20),
        "white_noise_25": AddWhiteNoise(max_snr_db = 25, min_snr_db=25),
        "env_noise_wind": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="wind", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_footsteps": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="footsteps", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_breathing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="breathing", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_coughing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="coughing", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_rain": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="rain", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_clock_tick": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu",  noise_category="clock_tick", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_sneezing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="sneezing", env_noise_dataset_path=env_noise_dataset_path),
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
        "freq_mask_03": FreqMask(prob=0.3),
        "freq_mask_05": FreqMask(prob=0.5),
        "fade_50_linear": AddFade(max_fade_size=0.5,fade_shape='linear', fix_fade_size=True),
        "fade_30_linear": AddFade(max_fade_size=0.3,fade_shape='linear', fix_fade_size=True),
        "fade_10_linear": AddFade(max_fade_size=0.1,fade_shape='linear', fix_fade_size=True),
        "fade_50_exponential": AddFade(max_fade_size=0.5,fade_shape='exponential', fix_fade_size=True),
        "fade_50_quarter_sine": AddFade(max_fade_size=0.5,fade_shape='quarter_sine', fix_fade_size=True),
        "fade_50_half_sine": AddFade(max_fade_size=0.5,fade_shape='half_sine', fix_fade_size=True),
        "fade_50_logarithmic": AddFade(max_fade_size=0.5,fade_shape='logarithmic', fix_fade_size=True),
        "resample_15000": ResampleAugmentation([15000]),
        "resample_15500": ResampleAugmentation([15500]),
        "resample_16500": ResampleAugmentation([16500]),
        "resample_17000": ResampleAugmentation([17000]),
    }
    
    selected_manipulation_key = "white_noise_20"
    selected_transform = manipulations[selected_manipulation_key]
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(asvspoof_2019_LA_train_dataloader, model, encoder, criterion, optimizer, epoch, args, cut_length, selected_transform)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            # Rename the weight files according to each augmentation method
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "xlsr_state_dict": model.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="/white_noise/checkpoint_{:04d}.pth.tar".format(epoch),
            )

def train(asvspoof_2019_LA_train_dataloader, model, encoder, criterion, optimizer, epoch, args, cut_length, selected_transform=None,  augmentations_on_cpu=None, augmentations=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(asvspoof_2019_LA_train_dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.eval()
    encoder.train()
    end = time.time()
    
    for batch_idx, (audio_input, spks, labels) in enumerate(tqdm(asvspoof_2019_LA_train_dataloader)):
        data_time.update(time.time() - end)
        audio_input = audio_input.squeeze(1).to(device)
        labels = labels.to(audio_input.device)

        if augmentations_on_cpu is not None:
            audio_input = augmentations_on_cpu(audio_input)
        audio_input = audio_input.to(device)

        audio_length = audio_input.shape[-1]
        # Spoof audio augmentation
        if (labels == 0).any():
            spoof_audio = audio_input[labels == 0]
            batch_size = spoof_audio.size(0)
            augmented_audio_list = []

            for i in range(batch_size):
                if selected_transform is None:
                    transformed = spoof_audio[i]
                else:
                    transformed = selected_transform(spoof_audio[i].unsqueeze(0)).squeeze(0)

                if transformed.shape[-1] > audio_length:
                    transformed = transformed[..., :audio_length]
                elif transformed.shape[-1] < audio_length:
                    pad_size = audio_length - transformed.shape[-1]
                    transformed = F.pad(transformed, (0, pad_size))

                transformed = transformed.to(audio_input.device)
                augmented_audio_list.append(transformed)

            augmented_audio = torch.stack(augmented_audio_list)
            audio_input[labels == 0] = augmented_audio

        # check the length of the audio, if it is not the same as the cut_length, then repeat or clip it to the same length
        if audio_input.shape[-1] < cut_length:
            audio_input = audio_input.repeat(1, int(cut_length/audio_input.shape[-1])+1)[:, :cut_length]
        elif audio_input.shape[-1] > cut_length:
            audio_input = audio_input[:, :cut_length]

        # Forward
        with torch.no_grad():
            outputs = model(audio_input)
            xlsr_features = outputs.last_hidden_state
        out_stj, out_bldl = encoder(xlsr_features)

        loss_stj = criterion(out_stj, labels)
        loss_bldl = criterion(out_bldl, labels)
        final_loss = 0.5 * loss_stj + 0.5 * loss_bldl

        # Backward & Optimize
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        losses.update(final_loss.item(), audio_input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

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
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()