import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from torch.utils.data import DataLoader

class Augmentations:
    def __init__(self, rotation_degree=15, horizontal_flip_prob=0.5, vertical_flip_prob=0.5, brightness_factor=0.1, contrast_factor=0.1):
        self.rotation_degree=rotation_degree
        self.horizontal_flip_prob=horizontal_flip_prob
        self.vertical_flip_prob=vertical_flip_prob
        self.brightness_factor=brightness_factor
        self.contrast_factor=contrast_factor

    def __call__(self, image, mask):
        if random.random()>0.5:
            angle=random.uniform(-self.rotation_degree, self.rotation_degree)
            image=TF.rotate(image,angle)
            mask=TF.rotate(mask,angle)

        if random.random()<self.horizontal_flip_prob:
            image=TF.hflip(image)
            mask=TF.hflip(mask)

        if random.random()<self.vertical_flip_prob:
            image=TF.vflip(image)
            mask=TF.vflip(mask)

        if random.random()>0.5:
            brightness_factor=1+random.uniform(-self.brightness_factor, self.brightness_factor)
            image=TF.adjust_brightness(image,brightness_factor)

        if random.random()>0.5:
            contrast_factor=1+random.uniform(-self.contrast_factor, self.contrast_factor)
            image=TF.adjust_contrast(image,contrast_factor)
        
        return image,mask

def create_data_loaders(train_dataset,test_dataset,cfg,use_augmentation=True):
    if use_augmentation and hasattr(train_dataset,'transform'):
        augmentation=Augmentations()
        train_dataset.augmentation=augmentation
    train_loader=DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_worker, pin_memory=True, drop_last=True)

    test_loader=DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_worker, pin_memory=True)

    return train_loader,test_loader