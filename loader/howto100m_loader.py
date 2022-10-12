import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import json

class HT100M_DataLoader(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(self, csv, video_root='', caption_root='', min_time=4.0, fps=16, num_frames=16, size=224, crop_only=False, center_crop=True,
                benchmark=False, token_to_word_path='./data/dict.npy', max_words=20, num_candidates=1, num_clip=8, random_left_right_flip=False,):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = num_candidates
        self.random_flip = random_left_right_flip
        self.num_clip = num_clip

    def __len__(self):
        return len(self.csv)

    def _get_video(self, video_path, start, end):
        videos = th.zeros(self.num_clip, 3, self.num_frames, self.size, self.size)
        for i, (s, e) in enumerate(zip(start, end)):
            start_seek = random.randint(s, int(max(s, e - self.num_sec)))
            cmd = (
                ffmpeg
                .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
                .filter('fps', fps=self.fps)
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
            if self.random_flip and random.uniform(0, 1) > 0.5:
                cmd = cmd.hflip()
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
            videos[i] = video[:, :self.num_frames]
        return videos

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _get_text(self, caption):
        caption_json = open(caption, 'r')
        cap = pd.DataFrame(json.load(caption_json))
        start, end = [], []
        words = th.zeros(self.num_clip, self.max_words, dtype=th.long)
        if len(cap) < self.num_clip:
            for i in range(self.num_clip):
                start.append(int(cap['start'].values[min(i, len(cap)-1)]))
                end.append(int(cap['end'].values[min(i, len(cap)-1)]))
                words[i] = self.words_to_ids(cap['text'].values[min(i, len(cap)-1)])
        else:
            ind = random.randint(0, len(cap) - self.num_clip)
            for i in range(self.num_clip):
                start.append(int(cap['start'].values[ind + i]))
                end.append(int(cap['end'].values[ind + i]))
                words[i] = self.words_to_ids(cap['text'].values[ind + i])
        return words, start, end

    def __getitem__(self, idx):
        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.video_root, video_file)
        text, start, end = self._get_text(os.path.join(self.caption_root, video_id + '.json'))
        videos = self._get_video(video_path, start, end)
        return {'video': videos, 'text': text, 'start': th.tensor(start), 'end': th.tensor(end)}
