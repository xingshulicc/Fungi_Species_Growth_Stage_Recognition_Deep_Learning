# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:55:35 2022

@author: DELL
"""

import os
import csv

# get images path
home_path = os.path.join(os.path.expanduser('~'), 'Desktop')
home_path = os.path.join(home_path, 'fungi_research')
dir_path = os.path.join(home_path, '1st')

def Get_subdirs_path(path):
    subdir_names = os.listdir(path)
    subdirs_path = []
    for sub_dir in subdir_names:
        subdir_path = os.path.join(path, sub_dir)
        subdirs_path.append(subdir_path)
    
    return subdirs_path
    
time_dirs_path = Get_subdirs_path(dir_path)
# the length of time_dirs_path is: 4
Fourth_week_subdirs_path = Get_subdirs_path(time_dirs_path[3])
# the length of First_week_subdirs_path is: 19
Fourth_week_D_5_4_Files_path = Get_subdirs_path(Fourth_week_subdirs_path[18])

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

Fourth_week_D_5_4_Images_path = Check_image_file(Fourth_week_D_5_4_Files_path)

# generate corresponding labels
'''
Each image has two labels:
    First label is fungi species label  (0 ~ 18)
    Second label is time label (0 ~ 3)
'''
def Generate_labels(path):
    images_path_labels = []
    species_label = 12
    time_label = 3
    for img_path in path:
        images_path_labels.append([img_path, str(species_label), str(time_label)])
        
    return images_path_labels

Fourth_week_D_5_4_Images_path_Labels = Generate_labels(Fourth_week_D_5_4_Images_path)
# print(First_week_A_2_1_Images_path_Labels[0])

# write files' path and corresponding labels into a csv file
with open(os.path.join(os.getcwd(), 'Files_Labels.csv'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for element in Fourth_week_D_5_4_Images_path_Labels:
        writer.writerow(element)




