import cv2
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, transforms, mode, root_path, img_size = 2048):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.root_path = root_path
        self.img_size = img_size

    def __getitem__(self, index):
        lr_path = os.path.join(self.root_path, self.df['LR'].iloc[index]) 
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.resize(lr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        if self.mode == "train":
            hr_path = os.path.join(self.root_path, self.df['HR'].iloc[index]) 
            hr_img = cv2.imread(hr_path)
            if self.transforms is not None:
                transformed = self.transforms(image=lr_img, label=hr_img)
                lr_img = transformed['image'] / 255.
                hr_img = transformed['label'] / 255.
            
            return lr_img, hr_img
        else:
            file_name = lr_path.split('/')[-1]
            if self.transforms is not None:
                transformed = self.transforms(image=lr_img)
                lr_img = transformed['image'] / 255.
            return lr_img, file_name
        
    def __len__(self):
        return len(self.df)

def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )

def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )