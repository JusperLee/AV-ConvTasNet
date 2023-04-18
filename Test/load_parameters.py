'''
Author: Kai Li
Date: 2021-03-13 21:42:49
LastEditors: Kai Li
LastEditTime: 2021-03-14 21:06:45
'''
import sys
sys.path.append('../')
import torch
from model.av_model import AV_model
from config.config import parse


def load_state_dict_in(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if 'av_model' in k:
            update_dict[k[9:]] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    opt = parse('/home/likai/data2/AV-Project/Audio_Visual_Train/config/wujian/train.yml', is_train=False)
    pretrained_dict = torch.load('/home/likai/data2/AV-Project/Audio_Visual_Train/checkpoints/Wujian-Model-Baseline/epoch=115.ckpt',map_location='cpu')['state_dict']
    av_model = AV_model(**opt['AV_model'])
    for k, v in av_model.state_dict().items():
        print(k)
    print('---------------------------------------')
    model = load_state_dict_in(av_model, pretrained_dict)