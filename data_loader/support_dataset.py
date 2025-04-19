import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class SupportSet(Dataset):
    def __init__(self, root_dir, class_name, transform=None):
        self.class_path = os.path.join(root_dir, class_name)
        self.image_paths = [os.path.join(self.class_path, f)
                          for f in os.listdir(self.class_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image