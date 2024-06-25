import os
import sys
import numpy as np
import pickle
import nibabel as nib

current_file_path = os.path.abspath(__file__)

project_root = os.path.dirname(os.path.dirname(current_file_path))

sys.path.append(project_root)

from config import PediatricConfig, AdultConfig


config = AdultConfig()

modalities = config.modalities

train = config.BRATS_TRAIN
valid = config.BRATS_VALID


def load_data(file_name):
    assert os.path.exists(file_name), 'Invalid file name, check first.'

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def save(path, has_label=True, flag='-'):
    """
    The z-score normalization is applied but keep the background as zero, and save data with dtype=float32
    :param path: the path of target file for preprocessing
    :param has_label: training data or not
    :return: (height, width, depth, modal)
    """
    if has_label:
        label = np.array(load_data(path + flag + 'seg.nii.gz'), dtype='uint8')
        label[label == 4] = 3
    img = np.stack([np.array(load_data(path + flag + modal + '.nii.gz'), dtype='float32') for modal in modalities], -1)

    output = path + '-f32.pkl'
    mask = img.sum(-1) > 0
    for i in range(img.shape[-1]):
        channel = img[..., i]
        temp = channel[mask]

        channel[mask] = (temp - temp.mean()) / temp.std()

    with open(output, 'wb') as f:
        if has_label:
            pickle.dump((img, label), f)
        else:
            pickle.dump(img, f)

        print(output)


def process(info, flag='-'):
    root, has_label = info['dir'], info['has_label']
    profile = os.path.join(root, info['list'])
    dir_list = open(profile).read().splitlines()
    sample_list = [dir.split('/')[-1] for dir in dir_list]
    path_list = [os.path.join(root, dir, name) for dir, name in zip(dir_list, sample_list)]

    for path in path_list:
        save(path, has_label, flag)



if __name__ == '__main__':
    process(train, flag="_")

