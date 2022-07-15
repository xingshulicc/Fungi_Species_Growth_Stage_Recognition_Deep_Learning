# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul 15 13:10:31 2022

@author: DELL
"""
import os
import csv

# get file path
home_path = os.path.join(os.path.expanduser('~'), 'Desktop')
home_path = os.path.join(home_path, 'fungi_research')
file_name = 'test_set.txt'
file_path = os.path.join(home_path, file_name)

# read txt file and store elements into list
file_list = []
with open(file_path, 'r') as f:
    for line in f.readlines():
        file_list.append(list(line.split()))

# print(file_list[0][0])

new_root_path = os.getcwd()  # this is the new root path on the server
for element in file_list:
    subroot_path = element[0].split('\\', 4)[4]
    element[0] = os.path.join(new_root_path, subroot_path)
    
# print(file_list[0][0])

new_file_name = 'test_set.csv'
new_file_path = os.path.join(home_path, new_file_name)

with open(new_file_path, 'w', newline='') as f1:
    writer = csv.writer(f1, delimiter=',')
    for element in file_list:
        writer.writerow(element)


