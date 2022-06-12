
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import glob


class HMDB_DataLoader(Dataset):
    """HMDB Video-Text loader."""

    def __init__(
            self,
            data,
            video_root='',
            num_clip=4,
            num_frames=32,
            size=224,
            with_flip=True,
            crop_only=False,
            center_crop=True,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.data = pd.read_csv(data)
        self.video_root = video_root
        self.size = size
        self.num_frames = num_frames
        self.num_clip = num_clip
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.with_flip = with_flip
        self.label_dict = {'brush_hair': 0, 'cartwheel': 1, 'catch': 2, 'chew': 3, 'clap': 4, 'climb': 5, 'climb_stairs': 6, 'dive': 7, 'draw_sword': 8, 
                           'dribble': 9, 'drink': 10, 'eat': 11, 'fall_floor': 12, 'fencing': 13, 'flic_flac': 14, 'golf': 15, 'handstand': 16, 'hit': 17, 
                           'hug': 18, 'jump': 19, 'kick': 20, 'kick_ball': 21, 'kiss': 22, 'laugh': 23, 'pick': 24, 'pour': 25, 'pullup': 26, 'punch': 27, 
                           'push': 28, 'pushup': 29, 'ride_bike': 30, 'ride_horse': 31, 'run': 32, 'shake_hands': 33, 'shoot_ball': 34, 'shoot_bow': 35, 
                           'shoot_gun': 36, 'sit': 37, 'situp': 38, 'smile': 39, 'smoke': 40, 'somersault': 41, 'stand': 42, 'swing_baseball': 43, 
                           'sword': 44, 'sword_exercise': 45, 'talk': 46, 'throw': 47, 'turn': 48, 'walk': 49, 'wave': 50}


    def __len__(self):
        return len(self.data)

    def _get_video(self, video_path, num_clip, flip=False):
        cmd = (
            ffmpeg
            .input(video_path)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        output = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(0, video.shape[1] - self.num_frames, num_clip, dtype=int) 
        for i, s in enumerate(start_ind):
            output[i] = video[:, s:s+self.num_frames] 
        if flip:
            video = th.cat((output, th.flip(output, [4])), dim=0) 
        return output

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        label = self.data['label'].values[idx]
        split1 = self.data['split1'].values[idx]
        split2 = self.data['split2'].values[idx]
        split3 = self.data['split3'].values[idx]
        video_path = os.path.join(self.video_root, 'hmdb', label[:-5], video_id)
        if not(os.path.isfile(video_path)):
            raise ValueError
        video = self._get_video(video_path, self.num_clip, flip=self.with_flip)
        return {'video': video, 'label': self.label_dict[label[:-5]], 'split1': split1, 'split2': split2, 'split3': split3}

