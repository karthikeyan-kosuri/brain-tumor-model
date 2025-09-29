import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import warnings

class BrainMRIDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_exts=('.jpg', '.jpeg', '.png'), mask_exts=('.png', '.jpg'), image_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_exts = mask_exts
        self.image_size = image_size
        
        # Collect image files
        self.image_names = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(image_exts)
        ])
        
        if len(self.image_names) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
    
    def __len__(self):
        return len(self.image_names)
    
    def _find_mask_path(self, img_name):

        #Find corresponding mask for an image by trying all mask extensions.
        base_name = os.path.splitext(img_name)[0]
        for ext in self.mask_exts:
            mask_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.exists(mask_path):
                return mask_path
        
        # If no mask found, warn and return None
        warnings.warn(f"No mask found for {img_name}")
        return None
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = self._find_mask_path(img_name)
        
        try:
            # Load image
            image = Image.open(img_path).convert('L')
            
            # Load mask if available, else create blank mask
            if mask_path is not None:
                mask = Image.open(mask_path).convert('L')
            else:
                mask = Image.new('L', image.size)
                print(f"Warning: Using blank mask for {img_name}")
            
            # Resize to consistent size
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
            
            # Convert to tensor
            image = TF.to_tensor(image).float()
            mask = TF.to_tensor(mask).float()
            
            return image, mask
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_name}: {str(e)}")