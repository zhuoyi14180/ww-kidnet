import os
import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import ndimage
import math

np.random.seed(42)


def pkl_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        max_val = np.max(image)
        min_val = np.min(image)
        image = (image - max_val) / (max_val - min_val)

        return {'image': image, 'label': label}


class RandomFlip:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class RandomCrop:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}
    

class FixedCrop:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = 56
        W = 56
        D = 13

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class RandomIntensityShift:
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class RandomRotate:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Padding:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label} # (240,240,155) -> (240,240,160)


class ToTensor:
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2)) # (modal, height, width, depth)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform_train(sample):
    transform = transforms.Compose([
        Padding(),
        # RandomRotate(), # time consuming
        RandomCrop(),
        RandomFlip(),
        RandomIntensityShift(),
        ToTensor()
    ])

    return transform(sample)


def transform_valid(sample):
    transform = transforms.Compose([
        Padding(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return transform(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root, mode='train', split_rate=0.9):
        path_list, name_list = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                name_list.append(name)
                path = os.path.join(root, line, name + '-')
                path_list.append(path)
        indices = np.arange(len(path_list))
        np.random.shuffle(indices)

        path_list = [path_list[i] for i in indices]
        name_list = [name_list[i] for i in indices]

        self.mode = mode
        count = len(name_list)
        percent = round(count * split_rate)
        if mode == "train":
            self.name_list = name_list[:percent]
            self.path_list = path_list[:percent]
        elif mode == "valid":
            self.name_list = name_list[percent:]
            self.path_list = path_list[percent:]
        else:
            self.name_list = name_list
            self.path_list = path_list

    def __getitem__(self, item):
        path = self.path_list[item]
        if self.mode == 'train':
            image, label = pkl_load(path + 'f32.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_train(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkl_load(path + 'f32.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return (sample['image'], sample['label']), item
        else:
            image = pkl_load(path + 'f32.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image, item

    def __len__(self):
        return len(self.name_list)



