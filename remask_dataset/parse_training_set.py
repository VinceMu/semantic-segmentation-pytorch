"""
@usage: python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/ ./data/ADEChallengeData2016_remask/ -r
"""  
import cv2
import sys
import numpy as np
import os
import argparse
import json
from pathlib import Path

def to_single_channel(img):
    sliced = img[:,:,:1]
    return np.reshape(sliced, (sliced.shape[0],sliced.shape[1]))

def create_odgt_obj(original_root_dir, reparsed_image_path, height, width):
    odgt = {}
    img_name = reparsed_image_path.split("/")[-1].split(".")[0]
    reparsed_img_dir = "/".join(reparsed_image_path.split("/")[:-1])
    mode = reparsed_image_path.split("/")[-2] # .../trainig/img_name.png
    odgt["fpath_img"] = "{}/images/{}/{}.jpg".format(original_root_dir, mode, img_name).replace("data/","") # hack to account for ROOT_DIR in model config
    odgt["fpath_segm"] = "{}/{}.png".format(reparsed_img_dir, img_name).replace("data/","") # hack to account for ROOT_DIR in model config
    odgt["height"] = height
    odgt["width"] = width
    return odgt

def write_odgt(odgt_filepath, odgt_array):
    with open(odgt_filepath, 'w') as odgt_file:
        for odgt in odgt_array:
            odgt_string = json.dumps(odgt)
            odgt_file.write(odgt_string +"\n")


def remask_directory(original_root, original_dir, new_dir, remove_non_floor):
    odgt_array = []
    for filename in os.listdir(original_dir):
            if filename.endswith(".png"):
                path = os.path.join(original_dir, filename)
                img = cv2.imread(path)
                if remove_non_floor:
                    is_floor = np.isin(img, [4])
                    has_floor = np.any(is_floor)
                    if not has_floor:
                        print("skipping " + filename)
                        pass

                masked = np.where(img == 4, 2, 1).astype(np.uint8)
                new_path = os.path.join(new_dir, filename)
                cv2.imwrite(new_path, to_single_channel(masked))
                img_name = path.split("/")[-1].split(".")[0]
                odgt_array.append(create_odgt_obj(original_root, new_path, img.shape[0], img.shape[1]))
    return odgt_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("original_root_directory")
    parser.add_argument("new_root_directory")

    parser.add_argument("-r","--remove_nonfloor", action='store_true', default=False)
    args = parser.parse_args()

    remove_non_floor = args.remove_nonfloor

    new_root_directory = args.new_root_directory.replace("\\", "/")
    original_root_directory = args.original_root_directory.replace("\\", "/")

    train_directory = original_root_directory + "/annotations/training/"
    validation_directory = original_root_directory + "/annotations/validation/"

    new_train_directory = new_root_directory + "/annotations/training/"
    new_validation_directory = new_root_directory + "/annotations/validation/"

    odgt_training_path = new_root_directory + "/training_remasked.odgt"
    odgt_validation_path = new_root_directory + "/validation_remasked.odgt"

    Path(new_train_directory).mkdir(parents=True, exist_ok=True)
    Path(new_validation_directory).mkdir(parents=True, exist_ok=True)

    odgt_train = remask_directory(original_root_directory, train_directory, new_train_directory, remove_non_floor)
    write_odgt(odgt_training_path, odgt_train)

    odgt_validation = remask_directory(original_root_directory, validation_directory, new_validation_directory, remove_non_floor)
    write_odgt(odgt_validation_path, odgt_validation)
