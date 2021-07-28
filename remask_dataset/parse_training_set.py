"""
@usage: python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/ ./data/ADEChallengeData2016_remask/ ./data/combined_classes/ade20k_classes_66.csv 
"""
import cv2
import sys
import numpy as np
import os
import argparse
import json
from ade20k_combined.segmentation_class_combiner import SegmentationClassCombiner
from pathlib import Path

ADE20_NUM_CLASSES = 150


def to_single_channel(img):
    sliced = img[:, :, :1]
    return np.reshape(sliced, (sliced.shape[0], sliced.shape[1]))


def create_odgt_obj(original_root_dir, reparsed_image_path, height, width):
    odgt = {}
    img_name = reparsed_image_path.split("/")[-1].split(".")[0]
    reparsed_img_dir = "/".join(reparsed_image_path.split("/")[:-1])
    mode = reparsed_image_path.split("/")[-2]  # .../trainig/img_name.png
    odgt["fpath_img"] = "{}/images/{}/{}.jpg".format(
        original_root_dir, mode,
        img_name).replace("data/",
                          "")  # hack to account for ROOT_DIR in model config
    odgt["fpath_segm"] = "{}/{}.png".format(
        reparsed_img_dir,
        img_name).replace("data/",
                          "")  # hack to account for ROOT_DIR in model config
    odgt["height"] = height
    odgt["width"] = width
    return odgt


def write_odgt(odgt_filepath, odgt_array):
    with open(odgt_filepath, 'w') as odgt_file:
        for odgt in odgt_array:
            odgt_string = json.dumps(odgt)
            odgt_file.write(odgt_string + "\n")


def remask_directory(original_root, original_dir, new_dir, mapping_array):
    odgt_array = []

    for filename in os.listdir(original_dir):
        if filename.endswith(".png"):
            path = os.path.join(original_dir, filename)
            img = cv2.imread(path)

            # mapped = (mapping_func(img)).astype(np.uint8)
            mapped = mapping_array[img]

            new_path = os.path.join(new_dir, filename)
            cv2.imwrite(new_path, to_single_channel(mapped))
            odgt_array.append(
                create_odgt_obj(original_root, new_path, img.shape[0],
                                img.shape[1]))
    return odgt_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("original_root_directory")
    parser.add_argument("new_root_directory")
    parser.add_argument("mapping_filepath")

    args = parser.parse_args()

    new_root_directory = args.new_root_directory.replace("\\", "/")
    original_root_directory = args.original_root_directory.replace("\\", "/")
    mapping_filpath = args.mapping_filepath.replace("\\", "/")

    train_directory = original_root_directory + "/annotations/training/"
    validation_directory = original_root_directory + "/annotations/validation/"

    new_train_directory = new_root_directory + "/annotations/training/"
    new_validation_directory = new_root_directory + "/annotations/validation/"

    odgt_training_path = new_root_directory + "/training_remasked.odgt"
    odgt_validation_path = new_root_directory + "/validation_remasked.odgt"

    Path(new_train_directory).mkdir(parents=True, exist_ok=True)
    Path(new_validation_directory).mkdir(parents=True, exist_ok=True)

    class_mapping = SegmentationClassCombiner(mapping_filpath)
    indexer = np.array([
        class_mapping.get_combined_class_index(i)
        for i in range(0, ADE20_NUM_CLASSES + 1)
    ])

    # vectorized_mapping_func = np.vectorize(class_mapping.get_combined_class_index)

    odgt_train = remask_directory(original_root_directory, train_directory,
                                  new_train_directory, indexer)
    write_odgt(odgt_training_path, odgt_train)

    odgt_validation = remask_directory(original_root_directory,
                                       validation_directory,
                                       new_validation_directory, indexer)
    write_odgt(odgt_validation_path, odgt_validation)
