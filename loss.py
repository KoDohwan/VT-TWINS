import torch
import torch.nn as nn
from soft_dtw import SoftDTW
import numpy as np
from itertools import permutations

class S2DTW(torch.nn.Module):
    def __init__(self, args):
        super(S2DTW, self).__init__()
        self.args = args
        self.sdtw = SoftDTW(use_cuda=True, gamma=1e-1, dist_func='negative_dot')
        self.tda = TDA(self.args)
        
    def video_text(self, video_embd, text_embd):
        b, n, d = video_embd.shape
        pos = -self.sdtw(video_embd, text_embd)
        video_embd_row = video_embd.unsqueeze(0).expand(b, b, n ,d).reshape(-1, n ,d)
        text_embd_col = text_embd.unsqueeze(1).expand(b, b, n ,d).reshape(-1, n, d)
        neg = -self.sdtw(video_embd_row, text_embd_col).reshape(b, b)
        neg = torch.logsumexp(neg, 1)
        loss = torch.mean(neg - pos)
        return loss

    def forward(self, video_embd, text_embd):
        # video_embd, text_embd = self.tda(video_embd, text_embd)
        loss = self.video_text(video_embd, text_embd)
        return loss
    
class TDA(torch.nn.Module):
    def __init__(self, args):
        super(TDA, self).__init__()
        self.args = args
        self.num_clip = args.num_clip
        self.n = self.num_clip * self.num_clip
        self.perm = self.generate_permutations(self.num_clip).cuda()
        self.num_perm = self.perm.shape[0]
        self.softmin = nn.Softmin(dim=1)
        
    def negative_dot_product(self, x, y):
        z = torch.matmul(x, y.transpose(1, 2))
        return -z
        
    def check_temporal_condition(self, p):
        for i in range(len(p)):
            if abs(p[i] - i) > 2:
                return False
        return True
    
    def generate_permutations(self, num_clip):
        perm = permutations([i for i in range(num_clip)])
        temporal_condition_perm = []
        for p in perm:
            if self.check_temporal_condition(p):
                temporal_condition_perm.append(p)
        temporal_condition_perm = torch.tensor(temporal_condition_perm)
        return temporal_condition_perm
    
    def generate_distribution(self, embd):
        b = embd.shape[0]
        self_similarity = self.negative_dot_product(embd, embd).detach()
        self_similarity = self_similarity.unsqueeze(1)
        self.perm_ = self.perm.unsqueeze(0).unsqueeze(3).repeat(b, 1, 1, self.num_clip)
        perm_similarity = self_similarity.repeat(1, self.num_perm, 1, 1)
        perm_similarity = torch.gather(torch.gather(perm_similarity, 2, self.perm_), 3, self.perm_.transpose(2, 3))
        distribution = torch.norm(self_similarity - perm_similarity, p=2, dim=(2, 3))
        distribution = self.softmin(distribution * 50)
        distribution = torch.distributions.Categorical(distribution)
        return distribution
        
        
    def forward(self, video_embd, text_embd):
        b, d = video_embd.shape[0], video_embd.shape[2]
        distribution_video = self.generate_distribution(video_embd)
        distribution_text = self.generate_distribution(text_embd)
        self.perm_ = self.perm.unsqueeze(0).repeat(b, 1, 1)
        perm_video = distribution_video.sample().unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_clip)
        perm_video = torch.gather(self.perm_, 1, perm_video).squeeze(1)
        perm_text = distribution_text.sample().unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_clip)
        perm_text = torch.gather(self.perm_, 1, perm_text).squeeze(1)
        video_embd = torch.gather(video_embd, 1, perm_video.unsqueeze(2).repeat(1, 1, d))
        text_embd = torch.gather(text_embd, 1, perm_text.unsqueeze(2).repeat(1, 1, d))
        return video_embd, text_embd