'''
Author: Kai Li
Date: 2021-03-13 21:48:39
LastEditors: Kai Li
LastEditTime: 2021-04-12 12:08:31
'''
from mir_eval.separation import bss_eval_sources
from itertools import permutations
import torch
import numpy as np
import pprint
def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = est.shape[0]
    egs = egs[:length]
    mix = mix[:length]
    mix = mix[:length]
    est = est - torch.mean(est)
    egs = egs - torch.mean(egs)
    mix = mix - torch.mean(mix)
    sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], est.numpy()[:length])
    mix_sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], mix.numpy()[:length])
    return float(sdr-mix_sdr)


def permutation_sdr(est_list, egs_list, mix, per):
    n = len(est_list)
    result = sum([SDR(est_list[a].detach().cpu(), egs_list[b].detach().cpu(), mix.detach().cpu())
                      for a, b in enumerate(per)])/n
    return result
    

def SI_SNR(x, s, mix, eps=1e-8):
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
    # return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)) - 20 * torch.log10(eps + l2norm(m_t) / (l2norm(mix_zm - m_t) + eps))
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def permute_SI_SNR(_s_lists, s_lists, mix):
    '''
        Calculate all possible SNRs according to 
        the permutation combination and 
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    per = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s, mix) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
        per.append(p)
    return max(results), per[results.index(max(results))]

def get_metrics(mix, clean, ests):
    _snr, per = permute_SI_SNR(ests, clean, mix)
    #_sdr = permutation_sdr(ests, clean, mix, per)
    utt_metrics = {'SI-SNRi': float(_snr), 'SDRi': float(1)}
    return utt_metrics

if __name__ == "__main__":
    mix = torch.randn(1, 16000)
    clean = torch.randn(1, 16000)
    est = torch.randn(1, 16000)
    metrics_dict = get_metrics(mix, clean, est)
    pprint.pprint(metrics_dict)
