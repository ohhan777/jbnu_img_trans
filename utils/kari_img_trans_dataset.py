import torch
from pathlib import Path
import numpy as np
import cv2
import os
import glob
import rasterio
import random

class KariImgTransDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        self.scale_factor = 2
        self.crop_size = (512,512)
        self.lr_gsd, self.lr_bits = 0.55, 14
        self.hr_gsd, self.hr_bits = 0.31, 11
        
        self.hr_files = glob.glob(os.path.join(self.root, 'WV3/*.tif'))
        self.num_files = len(self.hr_files)
        if train:
            self.lr_files = glob.glob(os.path.join(self.root, 'K3A/*.tif'))
            self.num_files = max(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        hr_file = self.hr_files[idx % len(self.hr_files)]
        hr_img = self.open_geotiff(hr_file)
        dshr_img, hr_img = self.hr_transform(hr_img)
        if self.train: # train and validation
            lr_file = self.lr_files[idx % self.num_files]
            lr_img = self.open_geotiff(lr_file)
            lr_img = self.lr_transform(lr_img)
            return dshr_img.copy(), lr_img.copy(), hr_img.copy() , hr_file
        else: # test
            return dshr_img.copy(), hr_img.copy(), hr_file


    def __len__(self):
        return self.num_files
    
    def open_geotiff(self, path, band=1):
        with rasterio.open(path) as src:
            data = src.read(band)
        assert data.all() != None, f'{path} is not found'
        data = data.astype(np.uint16)
        return data
    
    def hr_transform(self, img):
        h, w = img.shape[:2]

        # upsampling 
        new_h =  int(h * self.scale_factor * self.hr_gsd / self.lr_gsd)
        new_w =  int(w * self.scale_factor * self.hr_gsd / self.lr_gsd)

        img = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_CUBIC)
        # random cropping
        h = self.scale_factor * self.crop_size[0]
        w = self.scale_factor * self.crop_size[1]
        img = self.random_crop(img, (h, w))
        
        # downsampling 
        ds_img = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_CUBIC)
               
        # normalization
        img = img.astype(np.float32)/(2**self.hr_bits - 1)
        ds_img = ds_img.astype(np.float32)/(2**self.hr_bits - 1)        

        return np.expand_dims(ds_img, axis=0), np.expand_dims(img, axis=0)
        
    def lr_transform(self, img):
        # random crop
        h = self.crop_size[0]
        w = self.crop_size[1]
        img = self.random_crop(img, (h, w))
        # normalization
        img = img.astype(np.float32)/(2**self.lr_bits - 1)

        return np.expand_dims(img, axis=0)
    
    def random_crop(self, img, crop_size):
        h, w = img.shape
        img = self.pad_image(img, h, w, crop_size,
                               (0.0,))
        new_h, new_w = img.shape
        x = random.randint(0, new_w - crop_size[1])
        y = random.randint(0, new_h - crop_size[0])
        img = img[y:y+crop_size[0], x:x+crop_size[1]]

        return img

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=pad_value)

        return pad_image