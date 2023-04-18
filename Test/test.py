'''
Author: Kai Li
Date: 2021-03-13 21:33:31
LastEditors: Kai Li
LastEditTime: 2021-04-12 12:09:55
'''
import sys
sys.path.append('../')
from Data.datasets import LRS3mixDataset
import torch
import os
import argparse
from transformer import CenterCrop, ColorNormalize
from model.av_model import AV_model
from model.video_model import video
from config.config import parse
import tqdm
from pprint import pprint
import soundfile as sf
from metrics import get_metrics
from load_parameters import load_state_dict_in
from model.load_video_parameters import update_parameter
from scipy.io import wavfile
import numpy as np

class Separation():
    def __init__(self, yaml_path, model_path, gpuid):
        super(Separation, self).__init__()
        opt = parse(yaml_path, is_train=False)
        self.n_src = opt['data']['n_src']
        self.device = torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid = tuple(gpuid)
        self.datasets = LRS3mixDataset(opt['data']['test_dir'], n_src=opt['data']['n_src'],
                               sample_rate=opt['data']['sample_rate'], segment=opt['data']['segment'])
        
        video_model = video(**opt['video_model'])
        pretrain = torch.load(opt['video_checkpoint']['path'], map_location='cpu')['model_state_dict']
        self.video_model = update_parameter(video_model, pretrain).to(self.device)
        self.av_model = AV_model(**opt['AV_model'])
        dicts = torch.load(model_path, map_location='cpu')
        self.av_model = load_state_dict_in(self.av_model, dicts['state_dict']).to(self.device)

    def inference(self, file_path):
        sisnri = 0
        sdri = 0
        index = 0
        torch.no_grad().__enter__()
        self.video_model.eval()
        self.av_model.eval()
        for batch in self.datasets:
            mix = batch[0].unsqueeze(0)
            refs = batch[1]
            mouth = torch.from_numpy(batch[2]).unsqueeze(0).numpy()
            mouth = CenterCrop(mouth, (112, 112))
            mouth = ColorNormalize(mouth)
            B, D, H, W = mouth.shape
            # B x D x H x W -> B x C x D x H x W
            mouth = np.reshape(mouth, (B, 1, D, H, W))
            mouth = torch.from_numpy(mouth).type(
                torch.float32).to(self.device)
            mix = mix.to(self.device)
            if len(self.gpuid) != 0:
                mouth_emb = self.video_model(mouth.to(self.device))
                est = self.av_model(mix, mouth_emb)
                ref = [refs.to(self.device).squeeze(0)]
                est = [est.squeeze()]
                metric = get_metrics(mix.squeeze(0), ref, est)
                sisnri += metric['SI-SNRi']
                sdri += metric['SDRi']
                '''
                metric = get_metrics(egs, refs, ests)
                sisnri += metric['SI-SNRi']
                sdri += metric['SDRi']
                spks = [torch.squeeze(s.detach().cpu()) for s in ests]
                '''
            index += 1
            '''
            for s in spks:
                index += 1
                os.makedirs(file_path+'/s'+str(index), exist_ok=True)
                filename = file_path+'/s'+str(index)+'/'+key
                sf.write(filename, s, 8000)
            '''
            print('\r SI-SNRi: {}, SDRi: {}  {}/{}'.format(metric['SI-SNRi'], metric['SDRi'], index, len(self.datasets)), end="")
        pprint('SI-SNRi: {}, SDRi: {}'.format(sisnri /
                                                 (len(self.datasets)), sdri/(len(self.datasets))))
        pprint("Compute over {:d} utterances".format(len(self.datasets)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yaml', type=str, default='/home/likai/data2/AV-Project/config/wujian/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model_path', type=str, default='/home/likai/data2/AV-Project/checkpoints/Wujian-Model-Baseline/epoch=67.ckpt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='../result', help='save result path')
    args = parser.parse_args()
    gpuid = [int(i) for i in args.gpuid.split(',')]
    separation = Separation(args.yaml, args.model_path, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()
