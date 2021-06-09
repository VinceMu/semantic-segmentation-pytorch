import csv
import numpy as np

class SegmentationClassCombiner():
    
    def __init__(self, combined_classes_path, raw_classes_path):
        self.combined_class_list = self._initialise_combine_class_list(combined_classes_path)
        self.raw_classes = self._initialise_class_list(raw_classes_path)

    def combine_segmented_image(self, seg_img):
        return np.vectorize(self.get_combined_class_index)(seg_img)

    def get_combined_class_index(self, ade20k_index):
        ade20k_class = self.raw_classes[ade20k_index]
        for index, class_set in enumerate(self.combined_class_list):
            if ade20k_class in class_set:
                return index + 2 # non-nothing classes start from 2
        return 1

    def _initialise_combine_class_list(self, path, class_separator=","):
        combined_class_list = [] # list of sets of strings
        with open(path, "r") as class_groupings_file:
            for group in class_groupings_file:
                classes_in_group = group.strip().split(class_separator)
                combined_class_list.append(set(classes_in_group))
        return combined_class_list
    
    def _initialise_class_list(self, classes_path):
        names = []
        with open('data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]
        return names
