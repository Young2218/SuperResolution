import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import zipfile
from dataloader import CustomDataset, get_test_transform, get_train_transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
from models.SwinIR import SwinIR


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for split_imgs, file_name in tqdm(iter(test_loader)):

            pred = np.zeros((2048, 2048, 3))
            # print(len(split_imgs))
            for i, split_img in enumerate(split_imgs):
                # print("1:", split_img.shape)
                model.eval()
                split_img = split_img.to(device)
                pred_split_img = model(split_img)

                pred_split_img = pred_split_img.cpu().clone().detach().numpy()
                # print("2:", pred_split_img.shape)
                pred_split_img = pred_split_img[0].transpose(1, 2, 0)
                pred_split_img = pred_split_img*255.
                pred_split_img = pred_split_img.astype('uint8')

                x1 = i//8
                x2 = i % 8
                pred[256*x1:256*x1 + 256, 256*x2:256 *
                     x2 + 256, :] = pred_split_img
                # cv2.imshow("show", pred_split_img)
                # cv2.waitKey(0)

            pred_img_list.append(pred.astype('uint8'))
            name_list.append(file_name)

    return pred_img_list, name_list

def submission(pred_name_list, pred_img_list):

    os.makedirs('./submission', exist_ok=True)
    os.chdir("./submission/")
    sub_imgs = []

    for path, pred_img in tqdm(zip(pred_name_list, pred_img_list)):
        cv2.imwrite(path[0], pred_img)
        sub_imgs.append(path[0])
    submission = zipfile.ZipFile("../submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('Done.')


# Hyerparameter Setting
CFG = {
    'LR_SIZE': 64,
    'HR_SIZE': 256,
    'MODEL_LOAD_PATH': "/home/prml/Documents/ChanYoung/model_save/swinir.pt",
    'ROOT_PATH': "/home/prml/Documents/ChanYoung/",
}

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

test_df = pd.read_csv(CFG['ROOT_PATH'] + '/test.csv')
test_dataset = CustomDataset(
    test_df, get_test_transform(), "test", CFG['ROOT_PATH'])
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False)

net = SwinIR(upscale=4, in_chans=3, img_size=CFG['LR_SIZE'], window_size=8,
             img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
             mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

# net.load_state_dict(torch.load(CFG['MODEL_LOAD_PATH']), strict=True)
# load = torch.load(CFG['MODEL_LOAD_PATH'])   
# net.load_state_dict(torch.load(CFG['MODEL_LOAD_PATH']), strict=True)


loaded_state_dict = torch.load(CFG['MODEL_LOAD_PATH'])
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)

img_list, name_list = inference(net, test_loader, device)
submission(name_list, img_list)
