'''
@Author: Kai Li
@Date: 2020-07-21 20:49:08
LastEditors: Kai Li
LastEditTime: 2021-03-23 13:42:08
'''
import sys
sys.path.append('../')
from Loss.new_sisnr import Loss
import numpy as np
import warnings
from torchvision import transforms
from System.transformer import RandomCrop, CenterCrop, ColorNormalize, HorizontalFlip
from argparse import Namespace
import pytorch_lightning as pl
import torch
warnings.filterwarnings('ignore')


class System(pl.LightningModule):
    def __init__(self, av_model, video_model, optimizer, train_loss_func, val_loss_func,
                 train_loader=None, val_loader=None, scheduler=None, config=None):
        super(System, self).__init__()
        self.av_model = av_model
        self.video_model = video_model
        self.optimizer = optimizer
        # loss function
        self.train_loss_func = train_loss_func
        self.val_loss_func = val_loss_func
        # DataLoader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Optimizer
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        # Metrics
        self.validate_train_SISDRi = 0.
        self.validate_SISDRi = 0.

    def forward(self, inputs, mouth):
        with torch.no_grad():
            mouth_emb = self.video_model(mouth)
        # mouth_emb = torch.randn(4, 256, 100).to(inputs.device)
        return self.av_model(inputs, mouth_emb)
        # return self.test(torch.randn(4, 1, 32000, requires_grad=True).to(inputs.device))

    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, mouth, name = batch
        mouth = mouth.detach().cpu().numpy()
        loss = None
        if train:
            mouth = RandomCrop(mouth, (112, 112))
            mouth = ColorNormalize(mouth)
            mouth = HorizontalFlip(mouth)
            B, D, H, W = mouth.shape
            # B x D x H x W -> B x C x D x H x W
            mouth = np.reshape(mouth, (B, 1, D, H, W))
            mouth = torch.from_numpy(mouth).type(
                torch.float32).type_as(inputs)
            est_targets = self(inputs, mouth)
            loss = Loss(est_targets, targets, inputs)
        else:
            mouth = CenterCrop(mouth, (112, 112))
            mouth = ColorNormalize(mouth)
            B, D, H, W = mouth.shape
            # B x D x H x W -> B x C x D x H x W
            mouth = np.reshape(mouth, (B, 1, D, H, W))
            mouth = torch.from_numpy(mouth).type(
                torch.float32).type_as(inputs)
            est_targets = self(inputs, mouth)
            loss = Loss(est_targets, targets, inputs, improvement=True)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = {'loss': loss}
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.validate_train_SISDRi += avg_loss.item()
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb, train=False)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        lr = self.optimizer.param_groups[0]['lr']
        tensorboard_logs = {'val_loss': avg_loss, 'lr': lr}
        self.validate_SISDRi += avg_loss.item()
        return {'val_loss': avg_loss,
                'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Required by pytorch-lightning. """
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_epoch_end(self):
        self.logger.experiment.log_metric(
            'validate_SISDRi_mean', self.validate_SISDRi, step=self.current_epoch)
        self.logger.experiment.log_metric(
            'validate_train_SISDRi_mean', -self.validate_train_SISDRi, step=self.current_epoch)
        self.logger.experiment.log_metric(
            'learning_rate', self.optimizer.param_groups[0]['lr'], step=self.current_epoch)
        self.validate_train_SISDRi = 0.
        self.validate_SISDRi = 0.
