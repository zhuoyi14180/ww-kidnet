print((1, 0) + (2, 3, 4))

for i in range(0):
    print(i)


import nibabel as nib
import os
import numpy as np

from config import PediatricConfig, AdultConfig

config = AdultConfig()


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


path = os.path.join(config.BRATS_TRAIN['dir'], "HGG", 'BraTS19_2013_12_1', "BraTS19_2013_12_1_seg.nii.gz")


label = np.array(nib_load(path), dtype='uint8', order='C')

label[label == 4] = 3

print(label.shape)

print(np.sum(label == 0))
print(np.sum(label == 1))
print(np.sum(label == 2))
print(np.sum(label == 3))
print(np.sum(label == 4))
