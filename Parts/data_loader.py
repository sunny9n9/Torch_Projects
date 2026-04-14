from .imports import Dataset, DataLoader
from .imports import os
from .imports import cv2
from .imports import torch
from .imports import numpy as np
IMG_DIM = 512

__all__ = ['DatasetLoader']

class DatasetLoader(Dataset):
    # we inherit dataset wherein we need to define 2 functions-
    # len and getitem which is gonna be used by torch
    def __init__(self, image_dir, mask_dir, image_dim=(IMG_DIM, IMG_DIM)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_dim
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
            # it assumes the size of self.images == self.masks and all names are same
            # also no misssing names or anything of sorts.

    def __len__(self):
        return len(self.images) # lets make an images var to store em all

    def __getitem__(self, index):
        # to use this "index" i will need to make a list and parse index
        # hence the need to store all file names in a list
        image = cv2.imread(os.path.join(self.image_dir, self.images[index]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[index]), cv2.IMREAD_GRAYSCALE) # need to specify else it read everythign as BGR

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_AREA)
        
        try:
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            # transpose(2, 0, 1) cuz tf/opencv -> H W C
            # but torch C H W so old index 2 = new index 0 and so on ...

        except np.exceptions.AxisError as asx:
            # transpose only works on existing dimentions or "axes"
            image = image[None, ...] 
            # ... is used for automatic axis detection
            # makes it (1, H, W) if no channels

        mask = mask[None, :, :].astype(np.float32) / 255.0
        return torch.tensor(image), torch.tensor(mask)