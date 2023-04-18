'''
Author: Kai Li
Date: 2021-03-11 22:34:09
LastEditors: Kai Li
LastEditTime: 2021-04-04 11:43:28
'''
import sys
sys.path.append('../../')
import torch
from model.Wujian_Model.video_model import video



def update_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if 'front3D' in k:
            k = k.split('.')
            k_ = 'front3d.conv3d.'+k[1]+'.'+k[2]
            update_dict[k_] = v
        if 'resnet' in k:
            k_ = 'front3d.'+k
            update_dict[k_] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model

if __name__ == "__main__":
    pretrained_dict = torch.load('video_resnet18.pt',map_location='cpu')['model_state_dict']
    mynet = video(1, 64, 256, resnet_dim=256, kernel_size=3, repeat=5)
    v1 = 0
    v2 = 0
    for k, v in mynet.state_dict().items():
        v1 = v.clone()
        break
    print('---------------------------------------')
    model = update_parameter(mynet, pretrained_dict)
    for k, v in mynet.state_dict().items():
        v2 = v.clone()
        break
    print(v1[0][0][0][0])
    print(v2[0][0][0][0])