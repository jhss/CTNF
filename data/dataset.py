import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
import sys

class CIFAR10_C(Dataset):
    
    def __init__(self, path, c_type, transform = None):
        
        images = np.load(os.path.join(path, "{}.npy".format(c_type)))
        labels = np.load(os.path.join(path, "labels.npy"))
        
        concat_images = np.empty(shape = (1,32,32, 3))
        concat_labels = np.empty(shape = 1)
        for i in range(1, 6):
            concat_images = np.concatenate(
                             [concat_images, 
                              images[(i-1) * 10000 : (i-1) * 10000 + 500, ...]],
                              axis = 0)
            concat_labels = np.concatenate([concat_labels, 
                                labels[(i-1) * 10000 : (i-1) * 10000 + 500, ...]])

        self.images = concat_images[1:,...].astype(np.uint8)
        #self.images = images.astype(np.uint8)
        self.labels = concat_labels[1:,...]
        #self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img   = self.images[index, ...]
        label = self.labels[index, ...]

        if self.transform:
            img = self.transform(img)

        return (img, label)

class LSUN(Dataset):

    def __init__(self, path, transform = None):
 
        self.root_dir = path
        self.imgs = [io.imread(os.path.join(self.root_dir, "{}.jpg".format(i))) for i in range(10000)]
        self.imgs = np.stack(self.imgs, axis = 0)
        self.transform = transform

    def __len__(self):
       
        return len(self.imgs)

    def __getitem__(self, index):
       
        img   = self.imgs[index]

        if self.transform:
             img = self.transform(img)

        return img

class TINY(Dataset):

    def __init__(self, path, transform = None):

        self.root_dir = path
        self.imgs = []
        for i in range(10000):
            img = io.imread(os.path.join(self.root_dir, "{}.jpg".format(i)))

            if len(img.shape) == 2: continue
            self.imgs.append(img)

        self.imgs = np.stack(self.imgs, axis = 0)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img   = self.imgs[index]

        if self.transform:
            img = self.transform(img)

        return img 
