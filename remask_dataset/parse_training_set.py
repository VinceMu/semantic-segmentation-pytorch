  
import cv2
import sys
import numpy as np
import os
from pathlib import Path

def to_single_channel(img):
    sliced = img[:,:,:1]
    return np.reshape(sliced, (sliced.shape[0],sliced.shape[1]))

if __name__ == '__main__':
    train_directory = sys.argv[1]
    new_train_directory = sys.argv[2]

    Path(new_train_directory).mkdir(parents=True, exist_ok=True)


    for filename in os.listdir(train_directory):
        if filename.endswith(".png"):
            path = os.path.join(train_directory, filename)
            img = cv2.imread(path)
            masked = np.where(img == 4, 2, 1).astype(np.uint8)

            new_path = os.path.join(new_train_directory, filename)
            cv2.imwrite(new_path, to_single_channel(masked))
    