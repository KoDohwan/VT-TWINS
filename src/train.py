import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import random
import time
import glob
import sys
from tqdm import tqdm

root_path = os.getcwd()
sys.path.append(root_path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from s3dg import S3D
from args import get_args
from loader.howto100m_loader import HT100M_DataLoader
from loss import S2DTW
from utils import AllGather, get_cosine_schedule_with_warmup

allgather = AllGather.apply


def main():
    args = get_args()
    if args.verbose:
        print(args)
    assert args.eval_video_root != '' or not(args.evaluate)
    assert args.video_path != ''
    assert args.caption_root != ''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    args.multiprocessing_distributed = True
    args.evaluate = False
 
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.world_size = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        main_worker(args.gpu, args.world_size, args)



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.gpu,
        )
    # create model
    model = S3D(args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / args.world_size)
            args.batch_size_val = int(args.batch_size_val / args.world_size)
            args.num_thread_reader = int(args.num_thread_reader / args.world_size)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    train_dataset = HT100M_DataLoader(
        csv=args.train_csv,
        video_root=args.video_path,
        caption_root=args.caption_root,
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
        num_clip = args.num_clip,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )

    criterion = S2DTW(args)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momemtum)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, len(train_loader) * args.epochs)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not(os.path.isdir(checkpoint_dir)) and args.rank == 0:
        os.mkdir(checkpoint_dir)

    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size 
    log("Starting training loop for rank: {}, total batch size: {}".format(args.gpu, total_batch_size), args)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, scheduler, epoch, train_dataset, args)
        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, checkpoint_dir, epoch + 1
            )


def train(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        s_step = time.time()
        batch_loss = TrainOneBatch(model, optimizer, scheduler, sample_batch, criterion, epoch, args)
        d_step = time.time() - s_step
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(f"Epoch {epoch+1:d}, Elapsed Time: {d:.3f}, Epoch status: {args.batch_size * args.world_size * float(i_batch) / len(dataset):.4f}, \
                Training loss: {running_loss / args.n_display:.4f}, Learning rates: {optimizer.param_groups[0]['lr']:.6f}", args)
            running_loss = 0.0
            s = time.time()

def TrainOneBatch(model, opt, scheduler, data, loss_fun, epoch, args):
    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    text = data["text"].cuda(args.gpu, non_blocking=args.pin_memory)
    text = text.view(-1, text.shape[-1])
    video = video / 255.0
    video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
    opt.zero_grad()
    with torch.set_grad_enabled(True):
        video_embd, text_embd = model(video, text)
        video_embd = F.normalize(video_embd).view(-1, args.num_clip, video_embd.shape[1])
        text_embd = F.normalize(text_embd).view(-1, args.num_clip, text_embd.shape[1])
        if args.distributed:
            video_embd = allgather(video_embd, args)
            text_embd = allgather(text_embd, args)
        loss= loss_fun(video_embd, text_embd)
    loss.backward()
    opt.step()
    scheduler.step()
    return loss

def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))

def log(output, args):
    with open(os.path.join(os.path.dirname(__file__), 'log' , './log.txt'), "a") as f:
        f.write(output + '\n')

if __name__ == "__main__":
    main()
