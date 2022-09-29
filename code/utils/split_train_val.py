from email.errors import StartBoundaryNotFoundDefect
from importlib.resources import path
import cv2
import numpy as np
import os
import random
import string
from shutil import copyfile


def split_save_img(img_folder_path, img_file_name, img_save_path, patch_size: int = 512):
    img = cv2.imread(img_folder_path + img_file_name, cv2.IMREAD_COLOR)
    width, height, _ = img.shape

    # split img
    img_list = []
    for w in range(patch_size, width+1, patch_size):
        for h in range(patch_size, height+1, patch_size):
            img_list.append(img[w-patch_size:w, h-patch_size:h])

    # save img
    for i, small_img in enumerate(img_list):
        uniq = str(i)
        save_name = img_save_path + img_file_name[:-4] + "_" + uniq+'.png'

        while os.path.isfile(save_name):
            uniq += 'a'
            save_name = img_save_path + \
                img_file_name[:-4] + "_" + uniq + '.png'
        cv2.imwrite(save_name, small_img)


def get_file_names(folder_path):
    file_name_list = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_name_list


def rename_all_files(folder_path):
    file_name_list = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for fn in file_name_list:
        os.rename(os.path.join(folder_path, fn),
                  os.path.join(folder_path, "a" + fn))


save_path = "/home/prml/Documents/ChanYoung/SuperResolution/data"

dataset = {"/home/prml/Documents/ChanYoung/SuperResolution/data/dataset/dacon/train/hr/": "dacon",
           "/home/prml/Documents/ChanYoung/SuperResolution/data/dataset/urban100/Urban 100/X4 Urban100/X4/HIGH x4 URban100/": "urban100",
           "/home/prml/Documents/ChanYoung/SuperResolution/data/dataset/div2k/DIV2K_train_HR/DIV2K_train_HR/": "div2k",
           "/home/prml/Documents/ChanYoung/SuperResolution/data/dataset/div2k/DIV2K_valid_HR/DIV2K_valid_HR/": "div2k"
           }
train_n = 0
val_n = 0
for path, prefix in dataset.items():
    for file_name in get_file_names(path):
        isTrain = True if random.random() < 0.85 else False
        if isTrain:
            src_path = os.path.join(path, file_name)
            dst_path = os.path.join(
                save_path, "train", prefix + "_" + file_name)
            copyfile(src_path, dst_path)
            train_n +=1
        else:
            src_path = os.path.join(path, file_name)
            dst_path = os.path.join(save_path, "val", prefix + "_" + file_name)
            copyfile(src_path, dst_path)
            val_n += 1
print(train_n)
print(val_n)


# for folder_path in paths:
#     file_names = get_file_names(folder_path)
#     for file_name in file_names:
#         split_save_img(folder_path, file_name, save_path, patch_size=256)
#         print(file_name)

# rename_all_files("/home/prml/Documents/ChanYoung/div2k/DIV2K_train_HR/DIV2K_train_HR")
