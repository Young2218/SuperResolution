import os
import pandas as pd

path = "/home/prml/Documents/ChanYoung/ImageDataSet"

train_df = pd.DataFrame(columns=["HR"])
val_df = pd.DataFrame(columns=["HR"])

result = []
train_list = os.listdir(os.path.join(path, "train"))
for name in train_list:
    d = "./train/" + name
    result.append(d)
    
train_df = pd.DataFrame(result,columns=["HR"])   
train_df.to_csv(os.path.join(path, "train.csv"))
print(train_df.info())

result = []
val_list = os.listdir(os.path.join(path, "val"))
for name in val_list:
    d = "./val/" + name
    result.append(d)

val_df = pd.DataFrame(result, columns=["HR"])
val_df.to_csv(os.path.join(path, "val.csv"))
print(val_df.info())
