'''
Author: Kai Li
Date: 2021-03-11 22:28:54
LastEditors: Kai Li
LastEditTime: 2021-03-24 14:51:04
'''
import sys
sys.path.append('../../')
from config.config import parse
import torch.nn as nn
import torch
import math
from model.video_model import video
from model.load_video_parameters import update_parameter
# ----------Basic Part-------------


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


# ----------Audio Part-------------


class Encoder(nn.Module):
    '''
       Audio Encoder
       in_channels: Audio in_Channels is 1
       out_channels: Encoder part output's channels
       kernel_size: Conv1D's kernel size
       stride: Conv1D's stride size
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv = Conv1D(in_channels, out_channels,
                           kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
           x: [B, T]
           out: [B, N, T]
        '''
        x = self.conv(x)
        x = self.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class Audio_1DConv(nn.Module):
    '''
       Audio part 1-D Conv Block
       in_channels: Encoder's output channels
       out_channels: 1DConv output channels
       b_conv: the B_conv channels
       sc_conv: the skip-connection channels
       kernel_size: the depthwise conv kernel size
       dilation: the depthwise conv dilation
       norm: 1D Conv normalization's type
       causal: Two choice(causal or noncausal)
       skip_con: Whether to use skip connection
    '''

    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 dilation=1,
                 norm='gln',
                 causal=False,
                 skip_con=False):
        super(Audio_1DConv, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation*(kernel_size - 1)
                    )//2 if not causal else (dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=self.pad, dilation=dilation, groups=out_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)
        self.B_conv = nn.Conv1d(out_channels, b_conv, 1)
        self.Sc_conv = nn.Conv1d(out_channels, sc_conv, 1)
        self.causal = causal
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        # x: [B, N, T]
        out = self.conv1x1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        if self.causal:
            out = out[:, :, :-self.pad]
        out = self.prelu2(self.norm2(out))
        if self.skip_con:
            skip = self.Sc_conv(out)
            B = self.B_conv(out)
            # [B, N, T]
            return skip, B+x
        else:
            B = self.B_conv(out)
            # [B, N, T]
            return B+x

class Audio_Sequential(nn.Module):
    def __init__(self, repeats, blocks,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 norm='gln',
                 causal=False,
                 skip_con=False):
        super(Audio_Sequential, self).__init__()
        self.lists = nn.ModuleList([])
        self.skip_con = skip_con
        for r in range(repeats):
            for b in range(blocks):
                self.lists.append(Audio_1DConv(
                 in_channels=in_channels,
                 out_channels=out_channels,
                 b_conv=b_conv,
                 sc_conv=sc_conv,
                 kernel_size=kernel_size,
                 dilation=(2**b),
                 norm=norm,
                 causal=causal,
                 skip_con=skip_con))
    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.lists)):
                skip, out = self.lists[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.lists)):
                out = self.lists[i](x)
                x = out
            return x
class Video_1Dconv(nn.Module):
    """
    video part 1-D Conv Block
    in_channels: video Encoder output channels
    conv_channels: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual
    """

    def __init__(self,
                 in_channels,
                 conv_channels,
                 kernel_size,
                 dilation=1,
                 residual=True,
                 skip_con=True,
                 first_block=True
                 ):
        super(Video_1Dconv, self).__init__()
        self.first_block = first_block
        # first block, not residual
        self.residual = residual and not first_block
        self.bn = nn.BatchNorm1d(in_channels) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.bconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.sconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.dconv(y)
        else:
            y = self.dconv(x)
        # skip connection
        if self.skip_con:
            skip = self.sconv(y)
            if self.residual:
                y = y + x
                return skip, y
            else:
                return skip, y
        else:
            y = self.bconv(y)
            if self.residual:
                y = y + x
                return y
            else:
                return y


class Video_Sequential(nn.Module):
    """
    All the Video Part
    in_channels: front3D part in_channels
    out_channels: Video Conv1D part out_channels
    kernel_size: the kernel size of Video Conv1D
    skip_con: skip connection
    repeat: Conv1D repeats
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 skip_con=True,
                 repeat=5):
        super(Video_Sequential, self).__init__()
        self.conv1d_list = nn.ModuleList([])
        self.skip_con = skip_con
        for i in range(repeat):
            in_channels = out_channels if i else in_channels
            self.conv1d_list.append(
                    Video_1Dconv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        skip_con=skip_con,
                        residual=True,
                        first_block=(i == 0)))

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.conv1d_list)):
                skip, out = self.conv1d_list[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.conv1d_list)):
                out = self.conv1d_list[i](x)
                x = out
            return x


class Concat(nn.Module):
    """
    Audio and Visual Concatenated Part
    audio_channels: Audio Part Channels
    video_channels: Video Part Channels
    out_channels: Concat Net channels
    """

    def __init__(self, audio_channels, video_channels, out_channels):
        super(Concat, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        # project
        self.conv1d = nn.Conv1d(audio_channels + video_channels, out_channels, 1)

    def forward(self, a, v):
        """
        a: audio features, N x A x Ta
        v: video features, N x V x Tv
        """
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                                   a.size(1), v.size(1), self.audio_channels,
                                   self.video_channels))
        # up-sample video features
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        # concat: n x (A+V) x Ta
        y = torch.cat([a, v], dim=1)
        # conv1d
        return self.conv1d(y)
    
class AV_model(nn.Module):
    """
    Audio and Visual Speech Separation
    Audio Part
        N	Number of ﬁlters in autoencoder
        L	Length of the ﬁlters (in samples)
        B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        SC  Number of channels in skip-connection paths’ 1 × 1-conv blocks
        H	Number of channels in convolutional blocks
        P	Kernel size in convolutional blocks
        X	Number of convolutional blocks in each repeat   
    Video Part
        E   Number of ﬁlters in video autoencoder
        V   Number of channels in convolutional blocks
        K   Kernel size in convolutional blocks
        D   Number of repeats
    Concat Part
        F   Number of channels in convolutional blocks
    Other Setting
        R	Number of all repeats
        skip_con	Skip Connection
        audio_index     Number repeats of audio part
        norm    Normaliztion type
        causal  Two choice(causal or noncausal)
    """
    def __init__(
            self,
            # audio conf
            N=256,
            L=40,
            B=256,
            Sc=256,
            H=512,
            P=3,
            X=8,
            # video conf
            E=256,
            V=256,
            K=3,
            D=5,
            # fusion index
            F=256,
            # other
            R=4,
            skip_con=False,
            audio_index=2,
            norm="gln",
            causal=False):
        super(AV_model, self).__init__()
        self.video = Video_Sequential(E, V, K, skip_con=skip_con, repeat=D)
        # n x S > n x N x T
        self.encoder = Encoder(1, N, L, stride=L // 2)
        # before repeat blocks, always cLN
        self.cln = CumulativeLayerNorm(N)
        # n x N x T > n x B x T
        self.conv1x1 = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.skip_con = skip_con
        self.audio_conv = Audio_Sequential(
            audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        self.concat = Concat(B, V, F)
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # mask 1x1 conv
        # n x B x T => n x N x T
        self.mask = Conv1D(F, N, 1)
        # n x N x T => n x 1 x To
        self.decoder = Decoder(
            N, 1, kernel_size=L, stride=L // 2, bias=True)

    def check_forward_args(self, x, v):
        if x.dim() != 2:
            raise RuntimeError(
                "{} accept 1/2D tensor as audio input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if v.dim() != 3:
            raise RuntimeError(
                "{} accept 2/3D tensor as video input, but got {:d}".format(
                    self.__class__.__name__, v.dim()))
        if x.size(0) != v.size(0):
            raise RuntimeError(
                "auxiliary input do not have same batch size with input chunk, {:d} vs {:d}"
                .format(x.size(0), v.size(0)))

    def forward(self, x, v):
        """
        x: raw waveform chunks, N x C
        v: time variant lip embeddings, N x T x D
        """
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
            v = torch.unsqueeze(v, 0)
        # check args
        self.check_forward_args(x, v)

        # n x 1 x S => n x N x T
        w = self.encoder(x)
        # n x B x T
        a = self.conv1x1(self.cln(w))
        # audio feats: n x B x T
        a = self.audio_conv(a)
        # lip embeddings
        # N x T x D => N x V x T
        v = self.video(v)

        # audio/video fusion
        y = self.concat(a, v)

        # n x (B+V) x T
        y = self.feats_conv(y)
        # n x N x T
        m = torch.nn.functional.relu(self.mask(y))
        # n x To
        return self.decoder(w * m)

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

if __name__ == "__main__":
    frames = torch.randn(1, 1, 100, 96, 96)
    audio = torch.randn(1, 32000)
    opt = parse('/home/likai/data2/AV-Project/Audio_Visual_Train/config/wujian/train.yml')
    video_model = video(**opt['video_model'])
    pretrain = torch.load(opt['video_checkpoint']['path'], map_location='cpu')['model_state_dict']
    video_model = update_parameter(video_model, pretrain)
    v = video_model(frames)
    print(v.shape)
    av_model = AV_model(**opt['AV_model'])
    out = av_model(audio,v)
    print(out.shape)
    print(check_parameters(video_model), check_parameters(av_model))
    
    