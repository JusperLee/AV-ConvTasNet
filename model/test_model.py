'''
Author: Kai Li
Date: 2021-03-12 17:04:44
LastEditors: Kai Li
LastEditTime: 2021-03-12 17:10:01
'''
import torch.nn as nn
import torch
class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.linear1 = nn.Linear(32000, 100)
        self.linear2 = nn.Linear(100, 32000)

    def forward(self, x):
        '''
           x: [B, 1, T, H, W]
           out: [B, N, T]
        '''
        # [B, N, T]
        x = self.linear1(x)
        return self.linear2(x)

if __name__ == '__main__':
    a = torch.randn(4, 1, 32000)
    model = test()
    out = model(a)
    print(out.shape)