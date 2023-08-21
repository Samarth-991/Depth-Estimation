import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.load_tof_images import create_from_zip_absolute as load_tof_data

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384


class CreateAssignemntDataset(Dataset):
    def __init__(self, rgb_files, depth_files, transform=None, process_image=None, task='Train',image_shape=(384,384)):
        self.rgb_data = rgb_files
        self.depth_data = depth_files
        self.transform = transform
        self.pre_process = process_image
        self.reshape = image_shape

    def __len__(self):
        return len(self.depth_data)  # return the number of total samples contained in the dataset

    def __getitem__(self, index):
        rgb = self.rgb_data[index]
        depth = self.depth_data[index]
        calib_file = os.path.dirname(rgb).replace('rgb', 'calibration/0')

        data = load_tof_data(rgb_fpath=rgb, depthmap_fpath=depth, calibration_fpath=calib_file)
        rgb_image = data[8]
        depth_image = data[3]

        rgb_image= cv2.resize(rgb_image, self.reshape,interpolation = cv2.INTER_LINEAR)
        depth_image = cv2.resize(depth_image,self.reshape,interpolation = cv2.INTER_LINEAR)

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
