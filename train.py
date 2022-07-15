# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Mar 15 13:14:17 2022

@author: DELL
"""
import os

# before running the code, we should know the available GPU devices (ID)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

from Back_Bone import test_model
from Data_Batch import MyDataset
from torch.utils.data import DataLoader

# set GPU devices
device_ids = [0, 1]

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, )):
    # the shape of target is: [batch_size]
    # the shape of output is: [batch_size, classes]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # the shape of pred is: [maxk, batch_size]
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    # res contains top1 and top5 accuracy at the same time
    
    return res

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.getcwd(), 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    
    # switch to train mode
    model.train()
    
    start_time = time.time()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, sample_data in loop:
        data_time.update(time.time() - start_time)
        
        # loading data to GPUs
        img = sample_data['image'].cuda()
        label_1 = sample_data['species_label'].cuda()
        label_2 = sample_data['time_label'].cuda()
        
        # compute output and do SGD step
        optimizer.zero_grad()
        output = model(img)
        loss_1 = criterion(output['Species'], label_1)
        loss_2 = criterion(output['Time'], label_2)
        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()
        
        # measure accuracy (top_1) and record loss
        output = output.float()
        loss = loss.float()
        
        acc_1 = accuracy(output['Species'].data, label_1)[0]
        acc_2 = accuracy(output['Time'].data, label_2)[0]
        
        losses.update(loss.item(), img.size(0))
        
        top1_1.update(acc_1.item(), img.size(0))
        top1_2.update(acc_2.item(), img.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        
        # update progress bar
        loop.set_description(f'Epoch [{epoch}]')
        loop.set_postfix(loss=losses.avg, acc_1=top1_1.avg, acc_2=top1_2.avg, time=batch_time.avg)
        
    # save train accuracy and loss after each epoch
    log_folder = os.path.join(os.getcwd(), 'log')
    train_acc_1_f = os.path.join(log_folder, 'train_acc_1.txt')
    train_acc_2_f = os.path.join(log_folder, 'train_acc_2.txt')
    train_loss_f = os.path.join(log_folder, 'train_loss.txt')
    
    with open(train_acc_1_f, 'w') as f1:
        f1.write(str(top1_1.avg) + ' ')
    with open(train_acc_2_f, 'w') as f2:
        f2.write(str(top1_2.avg) + ' ')
    with open(train_loss_f, 'w') as f3:
        f3.write(str(losses.avg) + ' ')
    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    
    # switch to evaluate model
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for i, sample_data in enumerate(val_loader):
            img = sample_data['image'].cuda()
            label_1 = sample_data['species_label'].cuda()
            label_2 = sample_data['time_label'].cuda()
            
            output = model(img)
            loss_1 = criterion(output['Species'], label_1)
            loss_2 = criterion(output['Time'], label_2)
            loss = loss_1 + loss_2
            
            output = output.float()
            loss = loss.float()
            
            acc_1 = accuracy(output['Species'].data, label_1)[0]
            acc_2 = accuracy(output['Time'].data, label_2)[0]
            
            losses.update(loss.item(), img.size(0))
            top1_1.update(acc_1.item(), img.size(0))
            top1_2.update(acc_2.item(), img.size(0))
            
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            
    print(' * val_acc_1@1 {top1_1.avg:.4f}'.format(top1_1=top1_1))
    print(' * val_acc_2@1 {top1_2.avg:.4f}'.format(top1_2=top1_2))
    
    log_folder = os.path.join(os.getcwd(), 'log')
    val_acc_1_f = os.path.join(log_folder, 'val_acc_1.txt')
    val_acc_2_f = os.path.join(log_folder, 'val_acc_2.txt')
    val_loss_f = os.path.join(log_folder, 'val_loss.txt')
    
    with open(val_acc_1_f, 'w') as f4:
        f4.write(str(top1_1.avg) + ' ')
    with open(val_acc_2_f, 'w') as f5:
        f5.write(str(top1_2.avg) + ' ')
    with open(val_loss_f, 'w') as f6:
        f6.write(str(losses.avg) + ' ')
    
    return losses.avg 
    

def main():
    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-4
    
    batch_size = 16
    epochs = 300
    
    lowest_loss = 1e2
    
    # load model and train it on GPUs
    model = test_model()
    model = torch.nn.DataParallel(model, device_ids = device_ids)
    model.cuda()
    cudnn.benchmark = True
    
    # load data using DataLoader and perform data augmentation
    train_set_path = os.path.join(os.getcwd(), 'train_set.txt')
    test_set_path = os.path.join(os.getcwd(), 'test_set.txt')
    
    img_height = 256
    img_width = 256
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.Resize(size=(img_height, img_width)), 
        transforms.RandomCrop(size=224),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=(0, 30)), 
        transforms.RandomAutocontrast(p=0.5), 
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std), 
        ])
    val_transforms = transforms.Compose([
        transforms.Resize(size=(img_height, img_width)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std), 
        ])
    
    train_set = MyDataset(txt_path=train_set_path, transform=train_transforms)
    test_set = MyDataset(txt_path=test_set_path, transform=val_transforms)
    
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=4, 
                              pin_memory=True)
    val_loader = DataLoader(dataset=test_set, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True)
    
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), 
                                lr=lr, 
                                momentum=momentum, 
                                weight_decay=weight_decay, 
                                nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                        milestones=[50, 100, 150, 200, 250], 
                                                        gamma=0.125)
    
    # train model on train_set
    for epoch in range(1, epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()
        
        cur_loss = validate(val_loader, model, criterion)
        # remember best prec1 and save checkpoint
        is_best = cur_loss < lowest_loss
        lowest_loss = min(cur_loss, lowest_loss)
        save_checkpoint({
            'epoch': epoch, 
            'loss': cur_loss, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'lr_state_dict': lr_scheduler.state_dict(), 
            }, is_best, filename=os.path.join(os.getcwd(), 'checkpoint.pth.tar'))
        
        
    
    
    
if __name__ == '__main__':
    main()

   
  
  
  
  
   
    
       