import sys
from scipy.io import loadmat, savemat

save_path = sys.argv[1]

annots = loadmat('../data/color150.mat')
annots['colors'] = annots['colors'][3:4]
print(annots['colors'])
savemat(f"{save_path}/color_remask.mat", annots)
print(f"saved mats to {save_path}.color_remask.mat")
