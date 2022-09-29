import cv2
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, transforms, mode, root_path, hr_img_size = 512, lr_img_size = 128):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.root_path = root_path
        self.lr_img_size = lr_img_size
        self.hr_img_size = hr_img_size

    def __getitem__(self, index):
        
        if self.mode == "train" or self.mode == "val":
            # train or val mode learning with hr == 512*512, lr == 128*128
            hr_path = os.path.join(self.root_path, self.df['HR'].iloc[index])
            hr_img = cv2.imread(hr_path)
            hr_img = cv2.resize(hr_img, (self.hr_img_size, self.hr_img_size), interpolation=cv2.INTER_AREA)
            lr_img = cv2.resize(hr_img, (self.lr_img_size, self.lr_img_size), interpolation=cv2.INTER_AREA)
            #lr_img = cv2.resize(lr_img,(self.hr_img_size, self.hr_img_size), interpolation=cv2.INTER_CUBIC)

            if self.transforms:
                transformed = self.transforms(image=lr_img, label=hr_img)
                lr_img = transformed['image'] / 255.
                hr_img = transformed['label'] / 255.
            
            return lr_img, hr_img
        else: # self.mode == "test"
            # test mode learning with lr == 512*512 -> 64*64 -> divided 8
            # TODO: must check split img operates well

            lr_path = os.path.join(self.root_path, self.df['LR'].iloc[index]) 
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.resize(lr_img, (512, 512), interpolation=cv2.INTER_CUBIC)

            #TODO: make split img list
            split_imgs = []
            for i in range(0, 512, 64):
                for j in range(0, 512, 64):
                    split_imgs.append(lr_img[i:i+64, j:j+64])

            file_name = lr_path.split('/')[-1]
            split_tensors = []
            if self.transforms is not None:
                for s_img in split_imgs:
                    transformed = self.transforms(image=s_img)
                    split_tensors.append(transformed['image'] / 255.)

            return split_tensors, file_name
        
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