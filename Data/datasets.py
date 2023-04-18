'''
Author: Kai Li
Date: 2021-03-08 20:52:23
LastEditors: Kai Li
LastEditTime: 2021-04-06 20:39:37
'''
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import json

class LRS3mixDataset(Dataset):
    """Dataset class for the LRS3-mix source separation dataset.
    Args:
        json_dir (str): The path to the directory containing the json files.
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        n_src (int, optional): Number of sources in the training targets.
    """
    def __init__(self, json_dir, n_src=2, sample_rate=8000, segment=4.0):
        super().__init__()
        # Task setting
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.n_src = n_src
        self.like_test = self.seg_len is None
        # Load json files
        with open(json_dir, "r") as f:
            json_infos = json.load(f)
        # Filter out short utterances only when segment is specified
        orig_len = len(json_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(json_infos) - 1, -1, -1):  # Go backward
                if json_infos[i][3] < self.seg_len:
                    drop_utt += 1
                    drop_len += json_infos[i][2]
                    del json_infos[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )
        self.info = json_infos

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.info[idx][3] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.info[idx][3] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.info[idx][0], start=rand_start, stop=stop, dtype="float32")
        # Load sources
        source, _ = sf.read(self.info[idx][1], start=rand_start, stop=stop, dtype="float32")
        mouth = np.load(self.info[idx][2])['data']
        
        source = torch.from_numpy(source)    
        return torch.from_numpy(x), source, mouth, self.info[idx][0].split('/')[-1]

if __name__ == '__main__':
    dataset = LRS3mixDataset(json_dir='/home/likai/data2/AV-Project/Audio_Visual_Train/Data/wujian_data/cv.json')
    for idx in range(len(dataset)):
        print(dataset[idx][2])
        break