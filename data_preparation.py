# -*- coding: utf-8 -*-
"""
Created on 7/24/2020

@author: Senthil Pon Kumar P T, Bharath Shivapuram
@filename: data_preparation.py
@description: Light Tag
@comments: 
"""

import h5py
import os
import numpy as np
import cv2
from PIL import Image
import shutil
import json
from pathlib import Path

# input to feed the recording path to extract the images from h5 to jpg files
path_name = r"/home/uif41046/2021.06.21_at_07.45.31_camera-mi_680_mem-aff_4.rrec"

# output path to save 6FPS images
dest_pathname = r"/home/uif41046/extracted_images"

# sample h5 file name
h5_filename = r"/home/uif41046/2021.06.21_at_07.45.31_camera-mi_680_mem-aff_4.rrec/Input/{2021.06.21_at_07.45.31_camera-mi_680_mem-aff_4.rrec}_37153d9a_jpg.h5"
h5_recname = Path(h5_filename).stem

# json file for the h5 file
json_file = open("/home/uif41046/2021.06.21_at_07.45.31_camera-mi_680_mem-aff_4.rrec/Output/2021.06.21_at_07.45.31_camera-mi_680_mem-aff_4.rrec_SceneLabels.json", "r")
data = json.load(json_file)["Sequence"][0]["Labels"]

timestamps_with_construction = []
timestamps_without_construction = []

for frame in data:
    for label_name, labels in frame["Devices"][0]["Channels"][0]["SceneLabels"].items():
            if label_name == "Gen_Road_Works":
                for label in labels:
                    ts = label["starttimestamp"]
                    # Check if label contains "construction" in the category
                    if "Detection" in label["value"] or "Entering" in label["value"] or "Leaving" in label["value"]:
                        # Add timestamp to the list
                        timestamps_with_construction.append(int(ts))
            else:
                for label in labels:
                    ts = label["starttimestamp"]
                    timestamps_without_construction.append(ts)

def h5keys_to_image_conversion(h5_file_name, save_folder, fps_rate=15, count=0):

    h5_obj = h5py.File(h5_file_name, "r")
    for cam_channel in list(h5_obj.keys()):
        for data_channel in list(h5_obj[cam_channel].keys()):
            each_data = data_channel
            # for each_data in array_data:
            img_data = h5_obj[cam_channel][data_channel]
            # im = Image.fromarray(data)
            # im.save(os.path.join(save_folder, file_name))
            # if isinstance(img_data, h5py.Dataset):
            for ts, img_dataset in img_data.items():
            # if (count % fps_rate) == 0:
                if int(ts) in timestamps_with_construction:
                    file_name = f"construction/{h5_recname}_{ts}_export_{each_data}.jpg"
                    np_array = np.asarray(img_dataset, dtype=np.uint8)
                    np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    im = Image.fromarray(np_array[:, :, ::-1])
                    im.save(os.path.join(save_folder, file_name))
                #elif int(ts) in timestamps_without_construction:
                 #   file_name = f"nc_{ts}_export_{each_data}.jpg"
                elif (count % fps_rate) == 0:
                    file_name = f"not_construction/{h5_recname}_{ts}_export_{each_data}.jpg"
                    np_array = np.asarray(img_dataset, dtype=np.uint8)
                    np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    im = Image.fromarray(np_array[:, :, ::-1])
                    im.save(os.path.join(save_folder, file_name))

                count += 1


def h5_to_image_conversion(h5_file_name, save_folder, fps_rate=1, count=0):

    h5_obj = h5py.File(h5_file_name, "r")
    for cam_channel in list(h5_obj.keys()):
        for data_channel in h5_obj[cam_channel]:
            array_data = h5_obj[cam_channel][data_channel]
            for each_data in array_data:
                file_name = each_data + '.jpg'
                img_data = h5_obj[cam_channel][data_channel][each_data]
                # im = Image.fromarray(data)
                # im.save(os.path.join(save_folder, file_name))
                if isinstance(img_data, h5py.Dataset):
                    if (count % fps_rate) == 0:
                        np_array = np.asarray(img_data, dtype=np.uint8)
                        np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                        im = Image.fromarray(np_array[:, :, ::-1])
                        im.save(os.path.join(save_folder, file_name))
                    count += 1


def merge_short_to_long_folder(short_folder, long_folder):

    image_list = os.listdir(short_folder)
    no_files = len(image_list)
    print("total files: ", no_files)
    for file_name, count in zip(image_list, range(no_files)):
        short_file = os.path.join(short_folder, file_name)
        long_file = os.path.join(long_folder, file_name)
        shutil.copyfile(short_file, long_file)


def select_6fps_images(source_folder, destination_folder):

    image_list = os.listdir(source_folder)
    no_files = len(image_list)
    print("total files: ", no_files)
    for file_name, count in zip(image_list, range(no_files)):
        if (count % 5) == 0:
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copyfile(source_file, destination_file)


def single_stage_extraction(source_path, destination_path):

    long_str = "MFC5xx_long_image_right_h5_Undistorted"
    short_str = "MFC5xx_short_image_right_h5_Undistorted"
    name = source_path.split("\\")
    rec_name = name[len(name) - 1].split("_mfc5xx_Undistorted")[0]

    long_images = os.path.join(os.path.join(path_name, long_str), rec_name)
    print("Long image h5 extraction at 6FPS started...")
    print(long_images)
    h5_to_image_conversion(long_images + ".h5", destination_path, 5, 0)
    print("Long image h5 extraction at 6FPS completed")

    short_images = os.path.join(os.path.join(path_name, short_str), rec_name)
    print("Short image h5 extraction at 6FPS started...")
    print(short_images)
    h5_to_image_conversion(short_images + ".h5", destination_path, 5, -2)
    print("Short image h5 extraction at 6FPS completed")


def multi_stage_extraction(source_path):

    long_str = "MFC5xx_long_image_right_h5_Undistorted"
    short_str = "MFC5xx_short_image_right_h5_Undistorted"
    name = source_path.split("\\")
    rec_name = name[len(name)-1].split("_mfc5xx_Undistorted")[0]

    long_images = os.path.join(os.path.join(source_path, long_str), rec_name)
    print("Long image h5 extraction started...")
    print(long_images)
    h5_to_image_conversion(long_images + ".h5", long_images)
    print("Long image h5 extraction completed")

    short_images = os.path.join(os.path.join(source_path, short_str), rec_name)
    print("Short image h5 extraction started...")
    print(short_images)
    h5_to_image_conversion(short_images + ".h5", short_images)
    print("Short image h5 extraction completed")

    print("Short image merge to Long image started...")
    print(long_images)
    merge_short_to_long_folder(short_images, long_images)
    print("Short image merge to Long image completed")

    os.mkdir(long_images + '_6FPS')
    print("6FPS conversion started...")
    print(long_images + '_6FPS')
    select_6fps_images(long_images, long_images + '_6FPS')
    print("6FPS conversion finished...")
    print(".......All process completed ........")


def continuous_extraction(h5_file_name, save_folder):

    print("Continuous extraction process started")
    h5keys_to_image_conversion(h5_file_name=h5_filename,
                               save_folder=dest_pathname)
    print(".......Extraction process completed ........")


if __name__ == '__main__':

    # single_stage_extraction(source_path=path_name,
    #                         destination_path=dest_pathname)

    # multi_stage_extraction(source_path=path_name)

    continuous_extraction(h5_file_name=h5_filename,
                          save_folder=dest_pathname)
