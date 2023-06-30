import cv2
import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# define the input image dimensions
INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320


class CreateDataset(Dataset):
    def __init__(self, rgb_files, depth_files, transform=None, process_image=None, task='Train'):
        self.rgb_data = rgb_files
        self.depth_data = depth_files
        self.transform = transform
        self.pre_process = process_image

    def __len__(self):
        return len(self.rgb_data)  # return the number of total samples contained in the dataset

    def __getitem__(self, index):
        rgb = self.rgb_data[index]
        depth = self.depth_data[index]

        rgb_image = cv2.imread(rgb)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_image = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sample = self.transform(image=rgb_image, mask=depth_image)
            rgb_image, depth_image = sample['image'], sample['mask']

        if self.pre_process:
            sample = self.pre_process({'image': rgb_image, 'depth': depth_image})
            rgb_image, depth_image = sample['image'], sample['depth']
        return {'image': rgb_image, 'depth': depth_image}

    @staticmethod
    def _is_pil_image(img):
        return isinstance(img, Image.Image)

    @staticmethod
    def _is_numpy_image(img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
