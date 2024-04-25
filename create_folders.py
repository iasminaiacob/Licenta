import os
import numpy as np
import cv2
from PIL import Image
import shutil
from pathlib import Path
import random

path_c = "/home/uif41046/extracted_images/construction"
dest_train_c = "/home/uif41046/extracted_images/train/construction"
dest_val_c = "/home/uif41046/extracted_images/val/construction"

#path_nc = "/home/uif41046/extracted_images/not_construction"
#dest_train_nc = "/home/uif41046/extracted_images/train/not_construction"
#dest_val_nc = "/home/uif41046/extracted_images/val/not_construction"

c_imagefiles = [f for f in os.listdir(path_c) if os.path.isfile(os.path.join(path_c, f))]
#nc_imagefiles = [f for f in os.listdir(path_nc) if os.path.isfile(os.path.join(path_nc, f))]

num_images_train_c = int(len(c_imagefiles) * 0.8)
num_images_val_c = len(c_imagefiles) - num_images_train_c
#num_images_train_nc = int(len(nc_imagefiles) * 0.8)
#num_images_val_nc = len(nc_imagefiles) - num_images_train_nc

train_images_c = random.sample(c_imagefiles, num_images_train_c)
val_images_c = [img for img in c_imagefiles if img not in train_images_c]
#train_images_nc = random.sample(nc_imagefiles, num_images_train_nc)
#val_images_nc = [img for img in nc_imagefiles if img not in train_images_nc]

for img in train_images_c:
    shutil.move(os.path.join(path_c, img), os.path.join(dest_train_c, img))

for img in val_images_c:
    shutil.move(os.path.join(path_c, img), os.path.join(dest_val_c, img))

#for img in train_images_nc:
    #shutil.move(os.path.join(path_nc, img), os.path.join(dest_train_nc, img))

#for img in val_images_nc:
    #shutil.move(os.path.join(path_nc, img), os.path.join(dest_val_nc, img))