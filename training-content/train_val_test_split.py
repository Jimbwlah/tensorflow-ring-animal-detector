### Python script to split a labeled image dataset into Train, Validation, and Test folders.
# Author: Evan Juras, EJ Technology Consultants
# Date: 4/10/21

# Randomly splits images to 80% train, 10% validation, and 10% test, and moves them to their respective folders. 
# This script is intended to be used in the TFLite Object Detection Colab notebook here:
# https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb

import glob
from pathlib import Path
import random
import os

# Define paths to image folders
image_path = 'training-content/images/all'
train_path = os.path.join(os.getcwd(),'training-content/images/train').replace("\\", "/")
val_path = os.path.join(os.getcwd(),'training-content/images/validation').replace("\\", "/")
test_path = os.path.join(os.getcwd(),'training-content/images/test').replace("\\", "/")

# Get list of all images
jpg_file_list = [path for path in Path(image_path).rglob('*.jpg')]
png_file_list = [path for path in Path(image_path).rglob('*.png')]
bmp_file_list = [path for path in Path(image_path).rglob('*.bmp')]

file_list = jpg_file_list + png_file_list + bmp_file_list

file_num = len(file_list)
print('Total images: %d' % file_num)

# Determine number of files to move to each folder
train_percent = 0.9  # 80% of the files go to train
val_percent = 0.05 # 5% go to validation
test_percent = 0.05 # 5% go to test
train_num = int(file_num*train_percent)
val_num = int(file_num*val_percent)
test_num = file_num - train_num - val_num
print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)
print('Images moving to test: %d' % test_num)

def move_files(file_num, target_path):

    for i in range(file_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        move_me_safe = os.path.join(os.getcwd(),move_me).replace("\\", "/")
        parent_path = os.path.join(os.getcwd(),move_me.parent).replace("\\", "/")
        xml_fn = base_fn + '.xml'
        try:
            os.rename(os.path.join(parent_path,fn),os.path.join(target_path,fn))
            os.rename(os.path.join(parent_path,xml_fn),os.path.join(target_path,xml_fn))
        except:
            print("Error moving " + move_me_safe)
        file_list.remove(move_me)

# Select 80% of files randomly and move them to train folder
move_files(train_num, train_path)
move_files(val_num, val_path)
move_files(test_num, test_path)