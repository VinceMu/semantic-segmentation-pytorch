import csv
import numpy as np

class SegmentationClassCombiner():
    
    def __init__(self, combined_classes_path, raw_classes_path):
        self.class_mapping = self._initialise_class_mapping(combined_classes_path)
        self.raw_classes = self._initialise_class_list(raw_classes_path)

    def combine_segmented_image(self, seg_img):
        return np.vectorize(self.get_combined_class_index)(seg_img)
    
    def get_combined_class_index(self, ade20k_index):
        return self.class_mapping.get(ade20k_index, 1) # default 'nothing' class corrosponds to 1

    def _initialise_class_mapping(self, path, class_separator=","):
        idx_mapping = {} # map idx -> group
        with open(path, "r") as class_groupings_file:
            reader = csv.DictReader(class_groupings_file, delimeter =",", )
            for row in reader:
                key = row["idx"]
                val = row["group"]
                idx_mapping[key] = val 
        return idx_mapping

    def _initialise_class_list(self, classes_path):
        names = []
        with open('data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0] # only use the first name of ade20k classes
        return names
