from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import os
import random
from collections import OrderedDict
from train import train
from models.SRCNN import SRCNN
from models.EDSR import EDSR
from models.SwinIR import SwinIR
from models.HAT import HAT

# Hyerparameter Setting
CFG = {
    'LR_SIZE': 256,
    'HR_SIZE': 1024,
    'EPOCHS': 100000,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 1,
    'EARLY_STOP': 30,
    'MODEL_LOAD_PATH': "/home/prml/Documents/ChanYoung/SuperResolution/saved_model/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
    'MODEL_NAME': 'swinir',
    'ROOT_PATH': "/home/prml/Documents/ChanYoung/SuperResolution/data",
    'SAVE_PATH': "/home/prml/Documents/ChanYoung/SuperResolution/saved_model/"
}

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# set data loader
train_df = pd.read_csv(CFG['ROOT_PATH'] + '/train.csv')
val_df = pd.read_csv(CFG['ROOT_PATH'] + '/val.csv')

train_df = train_df[train_df['min_size'] >= 1024]
val_df = val_df[val_df['min_size'] >= 1024]

# train_df = train_df.iloc[:10]
# val_df = val_df.iloc[:1]

train_dataset = CustomDataset(
    train_df, "train", CFG['ROOT_PATH'], CFG['HR_SIZE'], CFG['LR_SIZE'])
train_loader = DataLoader(
    train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

val_dataset = CustomDataset(
    val_df, "val", CFG['ROOT_PATH'], CFG['HR_SIZE'], CFG['LR_SIZE'])
val_loader = DataLoader(
    val_dataset, 1, shuffle=False)


# swinir pre-trained -----------------------------------------------
net = SwinIR(upscale=4, in_chans=3, img_size=CFG['LR_SIZE'], window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
model = nn.DataParallel(net)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.L1Loss().to(device)





# EDSR ----------------------------------------------------------------------
# net = EDSR(n_feats=256, n_resblocks=32, res_scale=0.1, scale=4)
# # loaded_state_dict = torch.load(CFG['MODEL_LOAD_PATH']['state_dict'])

# model = nn.DataParallel(net)
# optimizer = torch.optim.Adam(
#     params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# criterion = nn.L1Loss().to(device)

# net.load_state_dict(torch.load(CFG['MODEL_LOAD_PATH']),strict=True)

# for param in net.parameters():
#     param.requires_grad = False


# model = nn.DataParallel(net)
# optimizer = torch.optim.Adam(
#     params=model.parameters(), lr=CFG["LEARNING_RATE"])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# criterion = nn.L1Loss().to(device)


best_model = train(model=model,
                   model_name=CFG['MODEL_NAME'],
                   train_loader=train_loader,
                   val_loader=val_loader,
                   optimizer=optimizer,
                   criterion=criterion,
                   scheduler=scheduler,
                   device=device,
                   save_path=CFG['SAVE_PATH'],
                   max_epoch=CFG['EPOCHS'],
                   early_stop=CFG['EARLY_STOP']
                   )
