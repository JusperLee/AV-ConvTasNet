'''
@Author: Kai Li
@Date: 2020-07-21 20:57:09
LastEditors: Kai Li
LastEditTime: 2021-03-23 13:32:10
FilePath: /Conv_TasNet_asteroid/train/train_sudo_rm_rf_gai.py
'''
import comet_ml
from pytorch_lightning.loggers import CometLogger
import sys
sys.path.append('../')
import os
import argparse
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config.config import parse
from model.video_model import video
from model.av_model import AV_model
from Data.datasets import LRS3mixDataset
from get_optimizer import make_optimizer
from System.system import System
from Loss.sisdr import PermInvariantSISDR
from model.load_video_parameters import update_parameter
import warnings
warnings.filterwarnings('ignore')


def main(conf):
    # default Dataset and Dataloader
    train_set = LRS3mixDataset(conf['data']['train_dir'], n_src=conf['data']['n_src'],
                               sample_rate=conf['data']['sample_rate'], segment=conf['data']['segment'])
    val_set = LRS3mixDataset(conf['data']['val_dir'], n_src=conf['data']['n_src'],
                             sample_rate=conf['data']['sample_rate'], segment=conf['data']['segment'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    # Define model and optimizer
    video_model = video(**conf['video_model'])
    pretrain = torch.load(conf['video_checkpoint']['path'], map_location='cpu')[
        'model_state_dict']
    video_model = update_parameter(video_model, pretrain)
    av_model = AV_model(**conf['AV_model'])
    optimizer = make_optimizer(av_model.parameters(), **conf['optim'])
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=conf['scheduler']['factor'],
                                      patience=conf['scheduler']['patience'], mode='max')
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf['root']
    # Define Loss function.
    train_loss_func = PermInvariantSISDR(batch_size=conf['training']['batch_size'],
                                         n_sources=1,
                                         zero_mean=True,
                                         backward_loss=True,)
    val_loss_func = PermInvariantSISDR(batch_size=conf['training']['batch_size'],
                                       n_sources=1,
                                       zero_mean=True,
                                       backward_loss=False,
                                       improvement=True,
                                       return_individual_results=True)

    system = System(av_model=av_model, video_model=video_model, train_loss_func=train_loss_func, val_loss_func=val_loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)

    # Define callbacks
    os.makedirs(os.path.join(exp_dir, 'checkpoints',
                             conf['training']['exp_name']), exist_ok=True)
    checkpoint_dir = os.path.join(
        exp_dir, 'checkpoints', conf['training']['exp_name'])
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='max', save_top_k=1, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       mode='max', verbose=1)
    # Don't ask GPU if they are not available.
    gpus = conf['gpus']

    # default logger used by trainer
    comet_logger = CometLogger(
        api_key='',
        save_dir=conf['log']['path'],  # Optional
        project_name=conf['log']['name'],  # Optional
        experiment_name=conf['training']['exp_name']  # Optional
    )
    comet_logger.log_hyperparams(conf)

    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_root_dir=exp_dir,
                         gpus=gpus,
                         distributed_backend='ddp',
                         train_percent_check=1.0,  # Useful for fast experiment
                         gradient_clip_val=5.,
                         logger=comet_logger,
                         )
    trainer.fit(system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    main(opt)
