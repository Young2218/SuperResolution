import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = "/home/prml/Documents/ChanYoung/ImageDataSet/split/"
train_save_path = "/home/prml/Documents/ChanYoung/train"
val_save_path = "/home/prml/Documents/ChanYoung/val"

train_num = 0
val_num = 0


for file_name in os.listdir(data_path):
    print(f"train num: {train_num}, val num: {val_num}")
    img = cv2.imread(os.path.join(data_path, file_name), cv2.IMREAD_COLOR)

    # train:val == 8:2
    isTrain = random.random()
    if isTrain < 0.8:
        cv2.imwrite(os.path.join(train_save_path,
                    str(train_num).zfill(6) + ".jpg"), img)
        train_num += 1

        flip_choices = random.sample([1, 2, 3, 4, 5, 6, 7], 2)
        for c in flip_choices:
            if c == 1:
                a_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif c == 2:
                a_img = cv2.rotate(img, cv2.ROTATE_180)
            elif c == 3:
                a_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif c == 4:
                a_img = cv2.flip(img, 1)
            elif c == 5:
                a_img = cv2.flip(img, 1)
                a_img = cv2.rotate(a_img, cv2.ROTATE_90_CLOCKWISE)
            elif c == 6:
                a_img = cv2.flip(img, 1)
                a_img = cv2.rotate(a_img, cv2.ROTATE_180)
            elif c == 7:
                a_img = cv2.flip(img, 1)
                a_img = cv2.rotate(a_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(os.path.join(train_save_path,
                        str(train_num).zfill(6) + ".jpg"), a_img)
            train_num += 1
    else:
        cv2.imwrite(os.path.join(val_save_path,
                    str(val_num).zfill(6) + ".jpg"), img)
        val_num += 1
