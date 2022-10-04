import cv2
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, mode, root_path, hr_size = 512, lr_size = 128):
        self.df = df
        self.mode = mode
        self.root_path = root_path
        self.lr_size = lr_size
        self.hr_size = hr_size

        if mode == "train":
            self.transforms = get_train_transform()
        else:
            self.transforms = get_test_transform()

    def __getitem__(self, index):
        
        if self.mode == "train" or self.mode == "val":
            # train or val mode learning with hr == 512*512, lr == 128*128
            hr_path = os.path.join(self.root_path, self.df['HR'].iloc[index])
            hr_img = cv2.imread(hr_path)
            hr_img = random_crop(hr_img, self.hr_size)
            
            lr_img = cv2.resize(hr_img, (0,0),fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            
            transformed = self.transforms(image=lr_img, label=hr_img)
            lr_img = transformed['image'] / 255.
            hr_img = transformed['label'] / 255.   
            
            return lr_img, hr_img
        
        else: # self.mode == "test"
            lr_path = os.path.join(self.root_path, self.df['LR'].iloc[index]) 
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.resize(lr_img, (512, 512), interpolation=cv2.INTER_CUBIC)

            
            split_imgs = []
            for i in range(0, 512, self.lr_size):
                for j in range(0, 512, self.lr_size):
                    split_imgs.append(lr_img[i:i+self.lr_size, j:j+self.lr_size])

            file_name = lr_path.split('/')[-1]
            split_tensors = []
            for s_img in split_imgs:
                transformed = self.transforms(image=s_img)
                split_tensors.append(transformed['image'] / 255.)

            return split_tensors, file_name
        
    def __len__(self):
        return len(self.df)
    
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(always_apply=True),
        ToTensorV2(p=1.0)
        ],
        additional_targets={'image': 'image', 'label': 'image'}
    )


def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )


def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]


