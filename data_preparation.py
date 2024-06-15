import h5py
import os
import numpy as np
import cv2
from PIL import Image
import shutil
import json
from pathlib import Path

# input
path_name = r"/home/uif41046/construction_site_dataset"

# output
dest_pathname = r"/home/uif41046/extracted_images"

#h5 file
h5_filename = r"" #redacted - confidential
h5_recname = Path(h5_filename).stem

# json file for the h5 file
json_file = open("", "r") #redacted - confidential
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
                        timestamps_with_construction.append(int(ts))
            else:
                for label in labels:
                    ts = label["starttimestamp"]
                    timestamps_without_construction.append(int(ts))

def h5keys_to_image_conversion(h5_file_name, save_folder, fps_rate=15, count=0):

    h5_obj = h5py.File(h5_file_name, "r")
    for cam_channel in list(h5_obj.keys()):
        for data_channel in list(h5_obj[cam_channel].keys()):
            each_data = data_channel
            img_data = h5_obj[cam_channel][data_channel]
            for ts, img_dataset in img_data.items():
                if int(ts) in timestamps_with_construction:
                    file_name = f"construction/{h5_recname}_{ts}_export_{each_data}.jpg"
                    np_array = np.asarray(img_dataset, dtype=np.uint8)
                    np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    im = Image.fromarray(np_array[:, :, ::-1])
                    im.save(os.path.join(save_folder, file_name))
                elif (count % fps_rate) == 0:
                    file_name = f"not_construction/{h5_recname}_{ts}_export_{each_data}.jpg"
                    np_array = np.asarray(img_dataset, dtype=np.uint8)
                    np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    im = Image.fromarray(np_array[:, :, ::-1])
                    im.save(os.path.join(save_folder, file_name))

                count += 1


def continuous_extraction(h5_file_name, save_folder):

    print("Continuous extraction process started")
    h5keys_to_image_conversion(h5_file_name=h5_filename,
                               save_folder=dest_pathname)
    print(".......Extraction process completed ........")


if __name__ == '__main__':
    continuous_extraction(h5_file_name=h5_filename,
                          save_folder=dest_pathname)
