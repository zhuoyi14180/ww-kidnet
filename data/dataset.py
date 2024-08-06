import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from .augment import Padding, RandomCrop2D, RandomCrop3D, RandomFlip2D, RandomFlip3D, RandomIntensityShift2D, RandomIntensityShift3D, RandomRotate3D, RandomRotate2D, ToTensor3D, ToTensor2D, FixedCrop


np.random.seed(42)


def pkl_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def transform_train(sample, mode_3d=True):
    if mode_3d:
        transform = transforms.Compose([
            Padding(),
            RandomRotate3D(), # time consuming
            RandomCrop3D(),
            RandomFlip3D(),
            RandomIntensityShift3D(),
            ToTensor3D()
        ])
    else:
        transform = transforms.Compose([
            RandomRotate2D(),
            RandomCrop2D((224, 224)), 
            RandomFlip2D(), 
            RandomIntensityShift2D(),
            ToTensor2D()
        ])

    return transform(sample)


def transform_valid(sample, mode_3d=True):
    if mode_3d:
        transform = transforms.Compose([
            Padding(),
            # MaxMinNormalization(),
            RandomCrop3D(), 
            ToTensor3D()
        ])
    else: 
        transform = transforms.Compose([
            RandomCrop2D((224, 224)),
            ToTensor2D()
        ])

    return transform(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root, mode='train', split_rate=0.9, lazy_load=True):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '-')
                paths.append(path)
        indices = np.arange(len(paths))
        np.random.shuffle(indices)

        paths = [paths[i] for i in indices]
        names = [names[i] for i in indices]

        self.mode = mode
        count = len(names)
        percent = round(count * split_rate)
        if mode == "train":
            self.names = names[:percent]
            self.paths = paths[:percent]
        elif mode == "valid":
            self.names = names[percent:]
            self.paths = paths[percent:]
        else:
            self.names = names
            self.paths = paths

        if not lazy_load:
            data = []
            for path in self.paths:
                data.append(pkl_load(path + 'f32.pkl'))
            self.data = data
        
        self.lazy_load = lazy_load
    

class BraTS3D(BraTS):
    def __init__(self, list_file, root, mode='train', split_rate=0.9, lazy_load=True):
        super().__init__(list_file, root, mode, split_rate, lazy_load)
        self.count = len(self.names)

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[item]
            sample = {'image': image, 'label': label}
            sample = transform_train(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[item]
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return (sample['image'], sample['label']), item
        else:
            image = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[item]
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image, item
        
    def __len__(self):
        return self.count
    

class BraTS2D(BraTS):
    def __init__(self, list_file, root, mode='train', split_rate=0.9, lazy_load=True):
        super().__init__(list_file, root, mode, split_rate, lazy_load)
        self.count = len(self.names) * 155

    def __getitem__(self, item):
        num = item // 155
        depth = item % 155

        path = self.paths[num]
        if self.mode == 'train':
            image, label = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[num]
            image = image[:, :, depth]
            label = label[:, :, depth]
            sample = {'image': image, 'label': label}
            sample = transform_train(sample, mode_3d=False)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[num]
            image = image[:, :, depth]
            label = label[:, :, depth]
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample, mode_3d=False)
            return (sample['image'], sample['label']), item
        else:
            image = pkl_load(path + 'f32.pkl') if self.lazy_load else self.data[num]
            image = image[:, :, depth]
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            image = torch.from_numpy(image).float()
            return image, item

    def __len__(self):
        return self.count