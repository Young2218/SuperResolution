from dataloader import CustomDataset, get_test_transform, get_train_transform
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import os
import random

from train import train
from models.SRCNN import SRCNN
from models.EDSR import EDSR
from models.SwinIR import SwinIR
from models.HAT import HAT

# Hyerparameter Setting
CFG = {
    'IMG_SIZE': 512,
    'EPOCHS': 500,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 1,
    'SEED': 41,
    'ROOT_PATH': "/home/prml/Documents/ChanYoung/ImageDataSet",
    'SAVE_PATH': "/home/prml/Documents/ChanYoung/model_save/swinir.pt"
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# set data loader
train_df = pd.read_csv(CFG['ROOT_PATH'] + '/train.csv')
val_df = pd.read_csv(CFG['ROOT_PATH'] + '/val.csv')
# test_df = pd.read_csv(CFG['ROOT_PATH'] + '/test.csv')

train_dataset = CustomDataset(
    train_df, get_train_transform(), "train", CFG['ROOT_PATH'])
train_loader = DataLoader(
    train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

val_dataset = CustomDataset(
    train_df, get_test_transform(), "val", CFG['ROOT_PATH'])
val_loader = DataLoader(
    val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# test_dataset = CustomDataset(
#     test_df, get_test_transform(), "test", CFG['ROOT_PATH'])
# test_loader = DataLoader(
#     test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# training
net = SwinIR(upscale=4, in_chans=3, img_size=CFG['IMG_SIZE'], window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')




model = nn.DataParallel(net)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.L1Loss().to(device)

best_model = train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   optimizer=optimizer,
                   criterion=criterion,
                   scheduler=scheduler,
                   device=device,
                   save_path=CFG['SAVE_PATH'],
                   max_epoch= CFG['EPOCHS'],
                   early_stop=10
                   )
