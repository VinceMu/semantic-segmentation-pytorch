"""
@usage: python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/annotations/training ./data/ADEChallengeData2016_remask/annotations/training ./data/ADEChallengeData2016_remask/training_remask.odgt -r
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

def create_odgt_obj(img_fullpath, new_train_directory, height, width):
    odgt = {}
    standardised_path = img_fullpath.replace("\\", "/")
    img_name = standardised_path.split("/")[-1].split(".")[0]
    odgt["fpath_img"] = "ADEChallengeData2016/images/training/{}.jpg".format(img_name)
    odgt["fpath_segm"] = "{}/{}.png".format(new_train_directory, img_name)
    odgt["height"] = height
    odgt["width"] = width
    return odgt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_directory")
    parser.add_argument("new_train_directory")
    parser.add_argument("odgt_file_path")
    parser.add_argument("-r","--remove_nonfloor", action='store_true', default=False)
    args = parser.parse_args()

    train_directory = args.train_directory
    new_train_directory = args.new_train_directory
    remove_non_floor = args.remove_nonfloor
    odgt_filepath = args.odgt_file_path
    odgt_directory = "".join(odgt_filepath.replace("\\", "/").split("/")[:-1])

    Path(new_train_directory).mkdir(parents=True, exist_ok=True)
    Path(odgt_directory).mkdir(parents=True, exist_ok=True)

    odgt_array = []

    for filename in os.listdir(train_directory):
        if filename.endswith(".png"):
            path = os.path.join(train_directory, filename)
            img = cv2.imread(path)
            if remove_non_floor:
                is_floor = np.isin(img, [4])
                has_floor = np.any(is_floor)
                if not has_floor:
                    print("skipping " + filename)
                    pass

            masked = np.where(img == 4, 2, 1).astype(np.uint8)
            new_path = os.path.join(new_train_directory, filename)
            cv2.imwrite(new_path, to_single_channel(masked))
            odgt_array.append(create_odgt_obj(path, new_train_directory, img.shape[0], img.shape[1]))
    
    with open(odgt_filepath, 'w') as odgt_file:
        for odgt in odgt_array:
            odgt_string = json.dumps(odgt)
            odgt_file.write(odgt_string +"\n")
