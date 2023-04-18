'''
Author: Kai Li
Date: 2021-03-12 09:52:53
LastEditors: Kai Li
LastEditTime: 2021-03-12 09:52:54
'''
import os
import yaml

def parse(opt_path, is_train=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES

    # is_train into option
    opt['is_train'] = is_train

    return opt
