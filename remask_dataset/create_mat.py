import sys
from scipy.io import loadmat, savemat

save_path = sys.argv[1]

annots = loadmat('../data/color150.mat')
annots['colors'] = annots['colors'][1:4]
print(annots)
savemat(f"{save_path}/color_remask.mat", annots)
print(f"saved mats to {save_path}.color_remask.mat")
