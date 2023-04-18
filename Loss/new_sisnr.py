'''
Author: Kai Li
Date: 2021-03-23 13:23:18
LastEditors: Kai Li
LastEditTime: 2021-03-23 13:27:54
'''
import torch

def sisnr(x, s, mix, improvement=False, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    
    length = x.shape[0]
    x = x[:length]
    s = s[:length]
    mix = mix[:length]
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
    
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    #-------------mix----------------
    m_t = torch.sum(
        mix_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    if improvement:
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)) - 20 * torch.log10(eps + l2norm(m_t) / (l2norm(mix_zm - m_t) + eps))
    else:
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def Loss(ests, egs, mix, improvement=False):
    # n x S
    l = len(egs)
    loss = sum(sisnr(ests, egs, mix, improvement))/l
    if improvement:
        return loss
    else:
        return -loss


if __name__ == "__main__":
    est = torch.randn(10,32000)
    egs = torch.randn(10,32000)
    print(Loss(est, est, egs, improvement=True))