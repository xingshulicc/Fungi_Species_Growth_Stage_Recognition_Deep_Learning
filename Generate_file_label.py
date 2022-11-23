# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:55:35 2022

@author: DELL
"""

import os
import csv
import numpy as np

# get images path
home_path = os.path.join(os.path.expanduser('~'), 'Desktop')
home_path = os.path.join(home_path, 'fungi_research')
dir_path = os.path.join(home_path, '1st')

def Get_probability(p):
    num = np.random.randint(low=0, high=100, size=(1,))[0]
    if num > (100 * p):
        return True
    else:
        return False


def Get_subdirs_path(path):
    subdir_names = os.listdir(path)
    subdirs_path = []
    for sub_dir in subdir_names:
        subdir_path = os.path.join(path, sub_dir)
        subdirs_path.append(subdir_path)
    
    return subdirs_path


# check file is image or not
def Check_image_file(path):
    img_files = []
    other_files = []
    for file_path in path:
        if file_path.endswith('.jpg'):
            img_files.append(file_path)
        else:
            other_files.append(file_path)
    
    return img_files


# generate corresponding labels
'''
Each image has two labels:
    First label is fungi species label  (0 ~ 18)
    Second label is time label (0 ~ 3)
'''
def Generate_labels(path, species_label, time_label):
    images_path_labels = []
    for img_path in path:
        images_path_labels.append([img_path, str(species_label), str(time_label)])
        
    return images_path_labels

time_dirs_path = Get_subdirs_path(dir_path)
proportion = 0.2
exclude_species = [9, 10, 11, 12, 13, 15]

train_images_path_labels = []
test_images_path_labels = []

for i, time in enumerate(time_dirs_path):
    species_list = Get_subdirs_path(time)
    for j, species in enumerate(species_list):
        if j in exclude_species:
            continue
        all_list = Get_subdirs_path(species)
        img_list = Check_image_file(all_list)
        for img_path in img_list:
            if Get_probability(proportion):
                train_images_path_labels.append([img_path, str(j), str(i)])
            else:
                test_images_path_labels.append([img_path, str(j), str(i)])


# write files' path and corresponding labels into a txt file
with open(os.path.join(os.getcwd(), 'train_set.txt'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    for element in train_images_path_labels:
        writer.writerow(element)

with open(os.path.join(os.getcwd(), 'test_set.txt'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    for element in test_images_path_labels:
        writer.writerow(element)


