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

# Hyerparameter Setting
CFG = {
    'IMG_SIZE': 2048,
    'EPOCHS': 100,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 1,
    'SEED': 41,
    'ROOT_PATH': "/home/rnjbdya/Downloads/open",
    'SAVE_PATH': "/home/rnjbdya/Downloads/open/model.pt"
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
test_df = pd.read_csv(CFG['ROOT_PATH'] + '/test.csv')

train_dataset = CustomDataset(
    train_df, get_train_transform(), "train", CFG['ROOT_PATH'])
train_loader = DataLoader(
    train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

test_dataset = CustomDataset(
    test_df, get_test_transform(), "test", CFG['ROOT_PATH'])
test_loader = DataLoader(
    test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# training
model = nn.DataParallel(EDSR())
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.MSELoss().to(device)

best_model = train(model=model,
                   train_loader=train_loader,
                   val_loader=None,
                   optimizer=optimizer,
                   criterion=criterion,
                   scheduler=scheduler,
                   device=device,
                   save_path=CFG['SAVE_PATH'],
                   max_epoch= CFG['EPOCHS'],
                   early_stop=10
                   )
