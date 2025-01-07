from utils.helpers import extract_images
from torch.utils.data import Dataset
import numpy as np

class MicroscopicImages(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Path to the rooth directory containing images
            transform (callable, optional): Optional transform to be applied on a sample.
            label (str or list): Target labels for the dataset, default is 'SVI'
        """
        self.root = root
        self.transform = transform
        # Use the extract_images_and_labels function to load images and labels
        self.images = extract_images(root) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
          try:
            image = self.transform(image)
          except Exception as e:
            print(f"Transform failed for image at index {idx}: {e}")
            return None, None

        return image

