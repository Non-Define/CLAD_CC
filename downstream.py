#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import json
from tqdm import tqdm

import torch
import torchaudio
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from aasist import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool, CONV, Residual_block, AasistEncoder
from model import DownStreamLinearClassifier
from datautils import  genSpoof_downstream_list, Dataset_ASVspoof2019, pad_or_clip_batch, AddWhiteNoise, VolumeChange, AddFade, WaveTimeStretch, PitchShift, CodecApply, AddEnvironmentalNoise, ResampleAugmentation, AddEchoes, TimeShift, AddZeroPadding
#-----------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Configurations
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)

with open("/home/cnrl/Workspace/ND/config.conf", "r") as f_json:
    config = json.load(f_json)

def load_model(config: dict):
    aasist_config_path = config['aasist_config_path']
    with open(aasist_config_path, "r") as f_json:
        aasist_config = json.load(f_json)
    
    return aasist_config["model_config"]

d_args = load_model(config)
encoder = AasistEncoder(d_args=d_args)

model_names = ["aasist_encoder"]
model_names = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Aasist Training")
parser.add_argument("data", metavar="DIR", nargs="?", default="/home/cnrl/Workspace/ND", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="aasist",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: aasist)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
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
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=30.0,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[60, 80],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by a ratio)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
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
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
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

parser.add_argument(
    "--pretrained", default="/home/cnrl/Workspace/ND/checkpoint/checkpoint_0149.pth.tar", type=str, help="path to moco pretrained checkpoint"
)
best_acc1 = 0

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
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args) -> None:
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args) -> None:
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        # pyre-fixme[61]: `print` is undefined, or not always defined.
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
    # create model
    # pyre-fixme[61]: `print` is undefined, or not always defined.
    print("=> creating model '{}'".format(args.arch))
    model = DownStreamLinearClassifier(encoder=encoder, input_depth=160)

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]

            new_state_dict = {}
            for k in list(state_dict.keys()):
                if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                    new_key = k[len("module.encoder_q."):]
                    new_state_dict[new_key] = state_dict[k]

            args.start_epoch = 0
            msg = encoder.load_state_dict(new_state_dict, strict=False)
            
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(
        parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # pyre-fixme[61]: `print` is undefined, or not always defined.
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # pyre-fixme[61]: `print` is undefined, or not always defined.
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            # pyre-fixme[61]: `print` is undefined, or not always defined.
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    database_path = config["database_path"]
    cut_length = 64600
    batch_size =12
    score_save_path = "./results/scores.txt"
    augmentations_on_cpu = None
    augmentations = None
    #---------------------------------------------------------------------------------------------------------------------------
    # train
    d_label_trn, file_train, utt2spk = genSpoof_downstream_list(
        json_path="/home/cnrl/Workspace/ND/train_label.json",
        is_train=False,
        is_eval=True
    )
    print('no. of ASVspoof 2019 LA downstreaming trials', len(file_train))

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
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )
    #---------------------------------------------------------------------------------------------------------------------------
    # validate
    d_label_trn, file_train, utt2spk = genSpoof_downstream_list(
        json_path="/home/cnrl/Workspace/ND/dev_label.json",
        is_train=False,
        is_eval=True
    )
    print('no. of ASVspoof 2019 LA downstreaming(validate) trials', len(file_train))
    
    asvspoof_LA_downstream_dataset = Dataset_ASVspoof2019(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(database_path, 'ASVspoof2019_LA_dev/'),
        cut_length=cut_length,
        utt2spk=utt2spk
    )
    asvspoof_2019_LA_downstream_dataloader = DataLoader(
        asvspoof_LA_downstream_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    noise_dataset_path = config["noise_dataset_path"]
    manipulations = {
        "no_augmentation": None,
        "volume_change_50": torchaudio.transforms.Vol(gain=0.5,gain_type='amplitude'),
        "volume_change_10": torchaudio.transforms.Vol(gain=0.1,gain_type='amplitude'),
        "white_noise_15": AddWhiteNoise(max_snr_db = 15, min_snr_db=15),
        "white_noise_20": AddWhiteNoise(max_snr_db = 20, min_snr_db=20),
        "white_noise_25": AddWhiteNoise(max_snr_db = 25, min_snr_db=25),
        "env_noise_wind": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="wind", noise_dataset_path=noise_dataset_path),
        "env_noise_footsteps": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="footsteps", noise_dataset_path=noise_dataset_path),
        "env_noise_breathing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="breathing", noise_dataset_path=noise_dataset_path),
        "env_noise_coughing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="coughing", noise_dataset_path=noise_dataset_path),
        "env_noise_rain": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="rain", noise_dataset_path=noise_dataset_path),
        "env_noise_clock_tick": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu",  noise_category="clock_tick", noise_dataset_path=noise_dataset_path),
        "env_noise_sneezing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="sneezing", noise_dataset_path=noise_dataset_path),
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
        "resample_15000": ResampleAugmentation([15000]),
        "resample_15500": ResampleAugmentation([15500]),
        "resample_16500": ResampleAugmentation([16500]),
        "resample_17000": ResampleAugmentation([17000]),
    }
    
    for batch_idx, (audio_input, spks, labels) in enumerate(tqdm(asvspoof_2019_LA_train_dataloader)):
        # audio_input = torch.squeeze(audio_input)
        audio_input = audio_input.squeeze(1)
        
        if augmentations_on_cpu != None:
            audio_input = augmentations_on_cpu(audio_input)
        audio_input = audio_input.to(device)

        audio_length = audio_input.shape[-1]
        mask = (labels != 0) & (labels != 1)
        if mask.any():
            spoof_audio = audio_input[mask]
            keys = list(manipulations.keys())
            random.shuffle(keys)

            batch_size = spoof_audio.size(0)
            augmented_audio_list = []

            for i in range(batch_size):
                key = keys[i % len(keys)]
                transform = manipulations[key]

                if transform is None:
                    transformed = spoof_audio[i]
                else:
                    transformed = transform(spoof_audio[i].unsqueeze(0)).squeeze(0)
                if transformed.shape[-1] > audio_length:
                    transformed = transformed[..., :audio_length]
                elif transformed.shape[-1] < audio_length:
                    pad_size = audio_length - transformed.shape[-1]
                    transformed = F.pad(transformed, (0, pad_size))
                    
                transformed = transformed.to(audio_input.device)
                augmented_audio_list.append(transformed)

            augmented_audio = torch.stack(augmented_audio_list)
            audio_input[mask] = augmented_audio

        # check the length of the audio, if it is not the same as the cut_length, then repeat or clip it to the same length
        if audio_input.shape[-1] < cut_length:
            audio_input = audio_input.repeat(1, int(cut_length/audio_input.shape[-1])+1)[:, :cut_length]
        elif audio_input.shape[-1] > cut_length:
            audio_input = audio_input[:, :cut_length]

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # pyre-fixme[16]: Optional type has no attribute `set_epoch`.
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(asvspoof_2019_LA_train_dataloader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(asvspoof_2019_LA_downstream_dataloader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)

def train(asvspoof_2019_LA_train_dataloader, model, criterion, optimizer, epoch, args) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(asvspoof_2019_LA_train_dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()  
    end = time.time()
      
    for i, batch in enumerate(asvspoof_2019_LA_train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        audio, ids, target = batch
        audio = audio.squeeze(1)
        
        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
            
        output = model(audio)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), audio.size(0))
        top1.update(acc1[0], audio.size(0))
        top5.update(acc5[0], audio.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(asvspoof_2019_LA_downstream_dataloader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(asvspoof_2019_LA_downstream_dataloader), [batch_time, losses, top1, top5], prefix="Test: "
    )
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(asvspoof_2019_LA_downstream_dataloader):
            audio, ids, target = batch
            audio = audio.squeeze(1)
            
            if args.gpu is not None:
                audio = audio.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(audio)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), audio.size(0))
            top1.update(acc1[0], audio.size(0))
            top5.update(acc5[0], audio.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(state, is_best, filename: str = "checkpoint.pth.tar") -> None:
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def sanity_check(state_dict, pretrained_weights) -> None:
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # name in pretrained model
        k_pre = (
            "module.encoder_q." + k[len("module.") :]
            if k.startswith("module.")
            else "module.encoder_q." + k
        )

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args) -> None:
    """Decay the learning rate based on schedule"""
    lr = args.lr
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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()