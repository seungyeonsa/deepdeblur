import os, glob, sys, numpy, random, math
import torch, torchvision
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
home_dir = '/mnt/home/20210846'

#Dataset

class GOPRODataset(Dataset):
    def __init__(self, train_path, train_ext, transform=None):
                
        files = glob.glob(os.path.join(home_dir, '%s/*/blur/*.%s'%(train_path, train_ext)))
        
        self.train_path = train_path
        self.train_ext = train_ext
        self.transform = transform
        self.data_list = []
        self.blur_list = glob.glob('data/GOPRO_Large/train/*/blur/*.png')
        self.sharp_list = [blur_path.replace('blur', 'sharp') for blur_path in self.blur_list]

        
        for lidx, file in enumerate(files):
            name = file.split('/')[-3]
            self.data_list.append(file)
            
    def __len__(self):
        return len(self.data_list) #2103 training images
    
    def __getitem__(self, idx):
        blur_path = self.blur_list[idx]
        sharp_path = self.sharp_list[idx]

        blur_image = Image.open(blur_path)
        sharp_image = Image.open(sharp_path)

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = GOPRODataset(
        train_path='./data/GOPRO_Large/train',
        train_ext='png',
        transform=transform
    )

    blur_sample, sharp_sample = train_set[0]
    print("Blur image shape:", blur_sample.shape)
    print("Sharp image shape:", sharp_sample.shape)
    