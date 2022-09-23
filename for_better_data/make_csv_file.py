import os
import pandas as pd

path = "/home/prml/Documents/ChanYoung"

max_train_num = -1
train_df = pd.DataFrame(columns=["HR"])
val_df = pd.DataFrame(columns=["HR"])


result = []
train_list = os.listdir(os.path.join(path, "train"))
for name in train_list:
    d = "./train/" + name
    result.append(d)


if max_train_num <= 0:
    train_df = pd.DataFrame(result,columns=["HR"])
else:
    train_df = pd.DataFrame(result[:max_train_num],columns=["HR"])
train_df.to_csv(os.path.join(path, "train.csv"))
print(train_df.info())

result = []
val_list = os.listdir(os.path.join(path, "val"))
for name in val_list:
    d = "./val/" + name
    result.append(d)

if max_train_num <= 0:
    val_df = pd.DataFrame(result, columns=["HR"])
else:
    max_val_num = max(1,max_train_num//4)
    val_df = pd.DataFrame(result[:max_val_num], columns=["HR"])
val_df.to_csv(os.path.join(path, "val.csv"))
print(val_df.info())
