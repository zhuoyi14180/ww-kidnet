from scipy import ndimage
import numpy as np
import random
import torch
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image


class MaxMinNormalization:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        max_val = np.max(image)
        min_val = np.min(image)
        image = (image - max_val) / (max_val - min_val)

        return {'image': image, 'label': label}


class RandomFlip3D:
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
    

class RandomFlip2D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)

        return {'image': image, 'label': label}


class RandomCrop3D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}
    

class RandomCrop2D:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        H = random.randint(0, 240 - 224)
        W = random.randint(0, 240 - 224)

        image = image[H: H + 224, W: W + 224, ...]
        label = label[..., H: H + 224, W: W + 224]

        return {'image': image, 'label': label}
    

class FixedCrop3D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = 56
        W = 56
        D = 13

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class FixedCrop2D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = 8
        W = 8

        image = image[H: H + 224, W: W + 224, ...]
        label = label[..., H: H + 224, W: W + 224]

        return {'image': image, 'label': label}


class RandomIntensityShift3D:
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class RandomIntensityShift2D:
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        scale_factor = np.random.uniform(1.0 - self.factor, 1.0 + self.factor)
        shift_factor = np.random.uniform(-self.factor, self.factor)
        image = image * scale_factor + shift_factor

        return {'image': image, 'label': label}


class RandomRotate3D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}
    

class RandomRotate2D:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, reshape=False)
        label = ndimage.rotate(label, angle, reshape=False)

        return {'image': image, 'label': label}


class Padding:
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label} # (240,240,155) -> (240,240,160)


class ToTensor3D:
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2)) # (modality, height, width, depth)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}
    

class ToTensor2D:
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(2, 0, 1)) # (modality, height, width)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}