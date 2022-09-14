import numpy as np
from torch import nn
from tqdm.auto import tqdm
import torch


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, save_path, max_epoch = 300, early_stop = 10):
    model.to(device)
    criterion.to(device)

    best_model = None
    best_loss = float('inf')
    check_early_stop = 0

    for epoch in range(1, max_epoch):
        # training
        model.train()
        train_loss = []
        for lr_img, hr_img in tqdm(iter(train_loader)):
            lr_img, hr_img = lr_img.float().to(device), hr_img.float().to(device)
            
            optimizer.zero_grad()
            
            pred_hr_img = model(lr_img)
            loss = criterion(pred_hr_img, hr_img)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        if scheduler is not None:
            scheduler.step()
        
        _train_loss = np.mean(train_loss)
        print(f'Epoch : [{epoch}] Train Loss : [{_train_loss:.5f}]')
        
        # validation
        if not val_loader:
            model.eval()
            val_loss = []
            for lr_img, hr_img in tqdm(iter(val_loader)):
                lr_img, hr_img = lr_img.float().to(device), hr_img.float().to(device)
            
                pred_hr_img = model(lr_img)
                loss = criterion(pred_hr_img, hr_img)

                val_loss.append(loss.item())
            
            _val_loss = np.mean(val_loss)
            print(f'Epoch : [{epoch}] Val Loss : [{_val_loss:.5f}]')

            if best_loss > _val_loss:
                best_loss = _val_loss
                best_model = model
                torch.save(model.state_dict(), save_path)
                check_early_stop = 0
                print("Minist Val Loss")
            else:
                # implement early stop
                check_early_stop += 1
                if check_early_stop > early_stop:
                    break
        else: # does not exist validation dataset
            if best_loss > _train_loss:
                best_loss = _train_loss
                best_model = model
                torch.save(model.state_dict(), save_path)
                check_early_stop = 0
                print("Minist Trian Loss")
            else:
                # implement early stop
                check_early_stop += 1
                if check_early_stop > early_stop:
                    break
            
    return best_model