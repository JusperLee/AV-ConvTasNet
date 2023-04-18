'''
Author: Kai Li
Date: 2021-03-23 13:03:46
LastEditors: Kai Li
LastEditTime: 2021-03-23 13:03:47
'''
import random
import cv2
import numpy as np


def CenterCrop(batch_img, size):
    '''
       Crop the center of image
       batch image: B x D x H x W (channel = 1, D = depth)
       size: (H x W)
    '''
    h, w = batch_img[0][0].shape[0], batch_img[0][0].shape[1]
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        w1 = int(round((w - tw))/2.)
        h1 = int(round((h - th))/2.)
        img[i] = batch_img[i, :, h1:h1+th, w1:w1+tw]
    return img


def RandomCrop(batch_img, size):
    '''
       Random Crop the center of image
       batch image: B x D x H x W
       size: (H x W)
    '''
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        h1 = np.random.randint(0, 8)
        w1 = np.random.randint(0, 8)
        img[i] = batch_img[i, :, h1:h1+th, w1:w1+tw]
    return img

def HorizontalFlip(batch_img):
    '''
        Random filp image by opencv
        batch image: B x D x H x W
    '''
    for i in range(len(batch_img)):
        if random.random()>0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j],1)
    return batch_img

def ColorNormalize(batch_img):
    '''
        Normal the image value
        batch image: B x D x H x W
    '''
    mean = 0.361858
    std = 0.147485
    batch_img = (batch_img - mean) / std
    return batch_img


if __name__ == "__main__":
    img = np.random.randn(10, 50, 120, 120)
    print(CenterCrop(img, (112, 112)).shape)
    print(RandomCrop(img, (112,112)).shape)