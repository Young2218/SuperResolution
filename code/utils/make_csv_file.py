import os
import pandas as pd
import cv2

path = "/home/prml/Documents/ChanYoung/SuperResolution/data"

train_df = pd.DataFrame(columns=["HR","min_size","dataset"])
val_df = pd.DataFrame(columns=["HR","min_size","dataset"])


result = []
train_list = os.listdir(os.path.join(path, "train"))
for name in train_list:
    hr = "./train/" + name
    dataset = name.split("_")[0]
    img = cv2.imread(os.path.join(path, hr), cv2.IMREAD_COLOR)
    w, h, _ = img.shape
    result.append([hr, min(w,h), dataset])

train_df = pd.DataFrame(result,columns=["HR","min_size","dataset"])

train_df.to_csv(os.path.join(path, "train.csv"))
print(train_df.info())

# --------- validation -------------------------------
result = []
val_list = os.listdir(os.path.join(path, "val"))
for name in val_list:
    hr = "./val/" + name
    dataset = name.split("_")[0]
    img = cv2.imread(os.path.join(path, hr), cv2.IMREAD_COLOR)
    w, h, _ = img.shape
    result.append([hr, min(w,h), dataset])

val_df = pd.DataFrame(result, columns=["HR","min_size","dataset"])

val_df.to_csv(os.path.join(path, "val.csv"))
print(val_df.info())
