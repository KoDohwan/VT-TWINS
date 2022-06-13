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
from loader.hmdb_loader import HMDB_DataLoader
from s3dg import S3D
from tqdm import tqdm
import numpy as np
import time
from utils import AllGather
from sklearn import preprocessing
from sklearn.svm import LinearSVC

allgather = AllGather.apply

def main(args):
    model = deploy_model(args)
    test_dataset = HMDB_DataLoader(data='./data/hmdb51.csv', num_clip=args.num_windows_test, video_root=args.eval_video_root,
                            num_frames=args.num_frames, size=args.video_size, crop_only=False, center_crop=True, with_flip=True, )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False, 
                                            num_workers=args.num_thread_reader, sampler=test_sampler)

    all_video_embd, labels, split1, split2, split3 = test(test_loader, model, args)
    if args.gpu == 0:
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        acc_list = []
        for reg in [100.0]:
            c = LinearSVC(C=reg)
            for split in range(3):
                if split == 0:
                    s = split1
                elif split == 1:
                    s = split2
                else:
                    s = split3
                X_train, X_test = all_video_embd[np.where(s == 1)[0]].reshape((-1, 1024)), all_video_embd[np.where(s == 2)[0]].reshape((-1, 1024))
                label_train, label_test = labels[np.where(s == 1)[0]].repeat(args.num_windows_test), labels[np.where(s == 2)[0]]
                print('Fitting SVM for split {} and C: {}'.format(split + 1, reg))
                c.fit(X_train, label_train)
                X_pred = c.decision_function(X_test)
                X_pred = np.reshape(X_pred, (len(label_test), args.num_windows_test, -1))
                X_pred = X_pred.sum(axis=1)
                X_pred = np.argmax(X_pred, axis=1)
                acc = np.sum(X_pred == label_test) / float(len(X_pred))  
                print("Top 1 accuracy split {} and C {} : {}".format(split + 1, reg, acc))
                acc_list.append(acc * 100)
        
        print('HMDB')
        print(f'Split1: {acc_list[0]:.2f} - Split2: {acc_list[1]:.2f} - Split3: {acc_list[2]:.2f} - Mean: {np.mean(acc_list):.2f}')
        with open('result.txt', 'a') as f:
            f.write('\nHMDB\n')
            f.write(f'Split1: {acc_list[0]:.2f} - Split2: {acc_list[1]:.2f} - Split3: {acc_list[2]:.2f} - Mean: {np.mean(acc_list):.2f}\n')

def test(test_loader, model, args):
    all_video_embd = []
    labels = []
    split1 = []
    split2 = []
    split3 = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_loader)):
            split1.append(data['split1'].cuda())
            split2.append(data['split2'].cuda())
            split3.append(data['split3'].cuda())
            labels.append(data['label'].cuda())
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd = model(video, None, mode='video', mixed5c=True)
            video_embd = video_embd.view(len(data['label']), -1, video_embd.shape[1])
            all_video_embd.append(video_embd)

    all_video_embd = torch.cat(all_video_embd, dim=0)
    all_video_embd = allgather(all_video_embd, args)
    labels = torch.cat(labels, dim=0)
    labels = allgather(labels, args)
    split1, split2, split3 = torch.cat(split1, dim=0), torch.cat(split2, dim=0), torch.cat(split3, dim=0)
    split1, split2, split3 = allgather(split1, args), allgather(split2, args), allgather(split3, args)
    return all_video_embd.cpu().numpy(), labels.cpu().numpy(), split1.cpu().numpy(), split2.cpu().numpy(), split3.cpu().numpy()
    

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
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, args))

if __name__ == "__main__":
    args = get_args()
    assert args.eval_video_root != ''
    spawn_workers(main, args)