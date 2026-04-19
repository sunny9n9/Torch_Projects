from .imports import Dataset, DataLoader
from .imports import os
from .imports import cv2
from .imports import torch
from .imports import numpy as np
from .imports import albumentations
IMG_DIM = 512

__all__ = ['DatasetLoader', 'DatasetLoaderV2']

class DatasetLoader(Dataset):
    # we inherit dataset wherein we need to define 2 functions-
    # len and getitem which is gonna be used by torch
    def __init__(self, image_dir, mask_dir, image_dim=(IMG_DIM, IMG_DIM), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_dim
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transformations = None
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

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

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

# well when augmenting data of medical field, we can't make stuff(s) look unrealistic, or actually that is the case for
# every data actually, the point is we have to be a bit more conservative on how much we stretch and compress images
# when augmenting data, we don't want model to learn absolute trash of organ structure, and we play around a bit more on
# the field of noise and brightness as not every machine producing scan is same, so these are two important things to keep in mind
# also the MASK must face the same changes as faced by input data, albumentaions(library) helps


class DatasetLoaderV2(Dataset):
    def __init__(self, data_dir, label_dir, dim, transformations=None):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.image_dim = dim
        self.transform = transformations

        self.data_files = sorted([f for f in os.listdir(data_dir) 
                                 if os.path.isfile(os.path.join(data_dir, f))])
        self.label_files = sorted([f for f in os.listdir(label_dir) 
                                  if os.path.isfile(os.path.join(label_dir, f))])

        self._basic_transform = albumentations.Compose([
            albumentations.Resize(height=dim[1], width=dim[0]),
            albumentations.Normalize(),
            albumentations.ToTensorV2()
        ])
        # whole albumetations library assumes we are using cv2/numpy style inputs, not tensors, so .ToTensorV2() should
        # be the absolute last step that we perform, if we do it in _basic_transform and then call some _further_transform
        # the _further_transform will get tensors as input, so call it at last, after user transforms
        

    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.data_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # normalizes fucking image only, wasted whole day on exploding gradient and negative loss
        # cuz it won't normalize masks implicitly
        basic_augmented = self._basic_transform(image=image, mask=mask)
        image, mask = basic_augmented['image'], basic_augmented['mask']
        mask = mask[None, ...] # cuz 2d(bw image)
        mask = mask.float() / 255.0
        
        return image, mask

    def __len__(self):
        return len(self.data_files)