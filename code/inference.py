import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import zipfile
from dataset import CustomDataset, get_test_transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
from models.SwinIR import SwinIR
from models.EDSR import EDSR


def inference(model, test_loader, device):
    model.to(device)

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

                x1 = i//4
                x2 = i % 4
                pred[512*x1:512*x1 + 512, 512*x2:512 *
                     x2 + 512, :] = pred_split_img
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
    'LR_SIZE': 128,
    'HR_SIZE': 512,
    'MODEL_LOAD_PATH': "/home/prml/Documents/ChanYoung/SuperResolution/saved_model/edsr_120_0.042.pt",
    'ROOT_PATH': "/home/prml/Documents/ChanYoung/SuperResolution/data"
}

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

test_df = pd.read_csv(CFG['ROOT_PATH'] + '/test.csv')
test_dataset = CustomDataset(
    test_df, "test", CFG['ROOT_PATH'])
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False)

net = EDSR(n_feats=256, n_resblocks=32, res_scale=0.1, scale=4)

# net.load_state_dict(torch.load(CFG['MODEL_LOAD_PATH']), strict=True)
# load = torch.load(CFG['MODEL_LOAD_PATH'])   
# net.load_state_dict(torch.load(CFG['MODEL_LOAD_PATH']), strict=True)


loaded_state_dict = torch.load(CFG['MODEL_LOAD_PATH'])
# new_state_dict = OrderedDict()
# for n, v in loaded_state_dict.items():
#     name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
#     new_state_dict[name] = v
# net.load_state_dict(new_state_dict)
net.load_state_dict(loaded_state_dict['state_dict'])
img_list, name_list = inference(net, test_loader, device)
submission(name_list, img_list)
