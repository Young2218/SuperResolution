import pandas as pd

train_df = pd.read_csv("/home/prml/Documents/ChanYoung/SuperResolution/data/train.csv")
val_df = pd.read_csv("/home/prml/Documents/ChanYoung/SuperResolution/data/val.csv")

print(train_df.describe())
print(val_df.describe())

# min = 564