import torch
import cv2
import numpy as np

im = cv2.imread("./data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png")
im2 = cv2.imread("./data/ADEChallengeData2016_remasked/annotations/training/ADE_train_00000001.png")
sliced = im2[:,:,:1]
reshaped = np.reshape(sliced, (sliced.shape[0],sliced.shape[1]))
print(reshaped.ndim)
print(reshaped.shape)

# print(im.shape)
# print(im2.shape)
# print(im[200][1])
# print(im2[1][1])
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
