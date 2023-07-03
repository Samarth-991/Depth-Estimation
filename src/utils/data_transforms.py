import albumentations as album
from torchvision import transforms
import numpy as np
import torch as th
import cv2

img_size = (480, 640)
BLUR_LIMIT = (3.0, 7.0)
SIGMA_LIMIT = 0


def get_training_augmentation():
    train_transform = [
        album.OneOf(
            [
                album.RandomBrightnessContrast(p=1),
                album.ChannelShuffle(p=1),
                album.Sharpen(p=1),
                album.GaussianBlur(p=1)
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)



class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)

        depth = cv2.resize(depth, (240, 320))
        depth = np.expand_dims(depth, axis=-1)
        depth = self.to_tensor(depth).float() * 1000
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = th.from_numpy(np.transpose(pic, (2, 0, 1)))
            return img.float().div(255)


def pre_process():
    return transforms.Compose([
        ToTensor(),
    ])


def pre_process_assignemnt():
    return transforms.Compose(
        [
            ToTensor(),

        ]

    )
