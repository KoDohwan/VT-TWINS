import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import random
import time
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from metrics import ctr
from args import get_args
from loader.crosstask_loader import CrossTask_DataLoader
from s3dg import S3D
from tqdm import tqdm
import numpy as np
import time
from utils import AllGather
allgather = AllGather.apply

def main(args):
    model = deploy_model(args)
    test_dataset = CrossTask_DataLoader(data='./data/crosstask.csv', num_clip=args.num_windows_test, video_root=args.eval_video_root, fps=args.fps,
                                        num_frames=args.num_frames, size=args.video_size, crop_only=False, center_crop=True, )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False, 
                                            num_workers=args.num_thread_reader, sampler=test_sampler)

    all_video_embd, all_text_embd, task_id = test(test_loader, model, args)
    if args.gpu == 0:
        video_dict = {}
        for i in range(len(task_id)):
            if task_id[i].item() not in video_dict.keys():
                video_dict[task_id[i].item()] = {}
                video_dict[task_id[i].item()]['video_embd'] = []
                video_dict[task_id[i].item()]['text_embd'] = []
            video_dict[task_id[i].item()]['video_embd'].append(all_video_embd[i])
            video_dict[task_id[i].item()]['text_embd'].append(all_text_embd[i])

        recall_list = []
        for task_id, videos in video_dict.items():
            all_video_embd = []
            all_text_embd = []
            for v, t in zip(videos['video_embd'], videos['text_embd']):
                all_video_embd.append(np.expand_dims(v, 0))
                all_text_embd.append(np.expand_dims(t, 0))
            all_video_embd = np.concatenate(all_video_embd, axis=0)
            all_text_embd = np.concatenate(all_text_embd, axis=0)
            similarity = np.dot(all_video_embd, all_text_embd.T)
            recall = ctr(similarity)
            recall_list.append(recall)
            
        print('CrossTask')
        print(f'CTR: {np.mean(recall_list):.2f}')
        with open('result.txt', 'a') as f:
            f.write('CrossTask\n')
            f.write(f'CTR: {np.mean(recall_list):.2f}\n')

def test(test_loader, model, args):
    all_text_embd = []
    all_video_embd = []
    video_id = []
    task_id = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_loader)):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            # video_id.append(data['video_id'].cuda())
            task_id.append(data['task_id'].cuda())
            
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            video_embd = F.normalize(video_embd).view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            text_embd = F.normalize(text_embd)
            all_video_embd.append(video_embd)
            all_text_embd.append(text_embd)

    all_video_embd, all_text_embd = torch.cat(all_video_embd, dim=0), torch.cat(all_text_embd, dim=0)
    all_video_embd, all_text_embd = allgather(all_video_embd, args), allgather(all_text_embd, args)
    task_id = torch.cat(task_id, dim=0)
    task_id = allgather(task_id, args)
    return all_video_embd.cpu().numpy(), all_text_embd.cpu().numpy(), task_id.cpu().numpy()
    

def deploy_model(args):
    checkpoint_path = args.pretrain_cnn_path
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    torch.cuda.set_device(args.gpu)
    model = S3D(args.num_class, space_to_depth=False, word2vec_path=args.word2vec_path)
    model.cuda(args.gpu)
    checkpoint_module = {k:v for k,v in checkpoint.items()}
    model.load_state_dict(checkpoint_module)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model.eval()
    
    print(f'Model Loaded on GPU {args.gpu}')
    return model

def main_worker(gpu, ngpus_per_node, main, args):
    cudnn.benchmark = True
    args.gpu = gpu
    args.rank = gpu
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=ngpus_per_node, rank=gpu)
    main(args)

def spawn_workers(main, args):
    ngpus_per_node = 8
    args.world_size = 8
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, args))

if __name__ == "__main__":
    args = get_args()
    args.num_windows_test = 1
    
    assert args.eval_video_root != ''
    spawn_workers(main, args)