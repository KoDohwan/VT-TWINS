import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import random
import socket
import time
import sys

root_path = os.getcwd()
sys.path.append(root_path)
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from metrics import retrieval
from args import get_args
from loader.msrvtt_loader import MSRVTT_DataLoader
from s3dg import S3D
from tqdm import tqdm
import numpy as np
import time
from utils import AllGather
allgather = AllGather.apply

def main(args):
    model = deploy_model(args)
    test_dataset = MSRVTT_DataLoader(data='./data/msrvtt_test.csv', num_clip=args.num_windows_test, video_root=args.eval_video_root, 
                                     fps=args.fps, num_frames=args.num_frames, size=args.video_size, crop_only=False, center_crop=True,)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False, 
                                            num_workers=args.num_thread_reader, sampler=test_sampler)

    all_video_embd, all_text_embd = test(test_loader, model, args)
    if args.gpu == 0:
        t2v = retrieval(np.dot(all_text_embd, all_video_embd.T))
        v2t = retrieval(np.dot(all_video_embd, all_text_embd.T))
        print('MSRVTT')
        print(f"R@1: {t2v['R1']:.2f} - R@5: {t2v['R5']:.2f} - R@10: {t2v['R10']:.2f} - Median R: {t2v['MR']}")
        print(f"R@1: {v2t['R1']:.2f} - R@5: {v2t['R5']:.2f} - R@10: {v2t['R10']:.2f} - Median R: {v2t['MR']}")
        with open('result.txt', 'a') as f:
            f.write('MSRVTT\n')
            f.write(f"R@1: {t2v['R1']:.2f} - R@5: {t2v['R5']:.2f} - R@10: {t2v['R10']:.2f} - Median R: {t2v['MR']}\n")
            f.write(f"R@1: {v2t['R1']:.2f} - R@5: {v2t['R5']:.2f} - R@10: {v2t['R10']:.2f} - Median R: {v2t['MR']}\n")

def test(test_loader, model, args):
    all_text_embd = []
    all_video_embd = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_loader)):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            video_embd = video_embd.view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            all_text_embd.append(text_embd)
            all_video_embd.append(video_embd)
    all_text_embd = torch.cat(all_text_embd, dim=0)
    all_video_embd = torch.cat(all_video_embd, dim=0)
    all_video_embd = allgather(all_video_embd, args)
    all_text_embd = allgather(all_text_embd, args)
    return all_video_embd.cpu().numpy(), all_text_embd.cpu().numpy()
    

def deploy_model(args):
    checkpoint_path = args.pretrain_cnn_path
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    torch.cuda.set_device(args.gpu)
    model = S3D(args.num_class, space_to_depth=False, word2vec_path=args.word2vec_path)
    model.cuda(args.gpu)
    checkpoint_module = {k[7:]:v for k,v in checkpoint.items()}
    model.load_state_dict(checkpoint_module)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model.eval()
    print(f'Model Loaded on GPU {args.gpu}')
    return model

def main_worker(gpu, ngpus_per_node, main, args):
    cudnn.benchmark = True
    args.gpu = gpu
    args.rank = gpu
    args.world_size = 8
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    args.dist_url = f'tcp://{ip}:12345'
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=ngpus_per_node, rank=gpu)
    main(args)

def spawn_workers(main, args):
    ngpus_per_node = 8
    args.world_size = 8
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, args))

if __name__ == "__main__":
    args = get_args()
    args.fps = 20
    args.num_windows_test = 8
    
    assert args.eval_video_root != ''
    spawn_workers(main, args)