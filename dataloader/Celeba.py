from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
from torch.utils import data
from torch.utils.data import Dataset

class Celeba(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.total_image = os.listdir(main_dir)
        
    def __len__(self):
        return len(self.total_image)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_image[idx])
        
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image

    

