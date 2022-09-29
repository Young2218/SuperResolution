import numpy as np
from torch import nn
from tqdm.auto import tqdm
import torch


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, save_path, max_epoch=300, early_stop=10):
    model.to(device)
    criterion.to(device)

    best_model = None
    best_loss = float('inf')
    check_early_stop = 0
    _train_loss = False
    _val_loss = False
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, max_epoch):
        # training
        model.train()
        
        train_loss_list.clear()
        for lr_img, hr_img in tqdm(iter(train_loader)):
            lr_img, hr_img = lr_img.float().to(device), hr_img.float().to(device)

            optimizer.zero_grad()

            pred_hr_img = model(lr_img)
            #print(f"hr: {hr_img.shape}, lr: {lr_img.shape}, pred: {pred_hr_img.shape}")
            loss = criterion(pred_hr_img, hr_img)

            loss.backward()
            optimizer.step()

            train_loss_list.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        _train_loss = np.mean(train_loss_list)

        # validation
        model.eval()
        val_loss_list.clear()
        with torch.no_grad():
            
            for lr_img, hr_img in tqdm(iter(val_loader)):
                lr_img, hr_img = lr_img.float().to(device), hr_img.float().to(device)

                pred_hr_img = model(lr_img)
                loss = criterion(pred_hr_img, hr_img)

                val_loss_list.append(float(loss.item()))

            _val_loss = np.mean(val_loss_list)

            if best_loss > _val_loss:
                best_loss = _val_loss
                best_model = model
                torch.save({"state_dict": model.module.state_dict(),
                            'loss': _val_loss,
                            'epoch': epoch},
                            save_path)
                check_early_stop = 0
                # print("Minist Val Loss")
            else:
                # implement early stop
                check_early_stop += 1
                if check_early_stop > early_stop:
                    break

        print(
            f'Epoch:{epoch}, Train Loss:{_train_loss:.5f}, Val Loss:{_val_loss:.5f}, es:{check_early_stop}')

    return best_model
