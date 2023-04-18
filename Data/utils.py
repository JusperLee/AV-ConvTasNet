'''
Author: Kai Li
Date: 2021-04-04 13:31:12
LastEditors: Kai Li
LastEditTime: 2021-04-04 13:31:20
'''
import torch

def data_nor(data, channel):
    mean = torch.mean(data, channel, keepdim=True)
    std = torch.std(data, channel, keepdim=True) + 1e-12
    nor_data = (data - mean) / std

    return nor_data, mean, std

def stft2spec(stft, normalized, save_phase, save_mean_std):
    magnitude = torch.norm(stft, 2, -1)

    if save_phase:
        # (1, 257, frames, 2) -> (257, frames, 2) -> (2, 257, frames)
        stft = stft.squeeze(0)
        stft = stft.permute(2, 0, 1)

        phase = stft / (magnitude + 1e-12)

        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), phase
        else:
            return (specgram, None, None), phase

    else:
        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), None
        else:
            return (specgram, None, None), None