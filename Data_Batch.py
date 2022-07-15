# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Mar  2 12:18:12 2022

@author: DELL
"""
from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], [int(words[1]), int(words[2])]))
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        
        label_1 = torch.tensor(data=label[0], dtype=torch.long)
        label_2 = torch.tensor(data=label[1], dtype=torch.long)
        
        img = Image.open(fn).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        sample = {
            'image': img, 
            'species_label': label_1, 
            'time_label': label_2
            }
            
        return sample
    
    def __len__(self):
        return len(self.imgs)


# please refer to: https://medium.com/jdsc-tech-blog/multioutput-cnn-in-pytorch-c5f702d4915f
