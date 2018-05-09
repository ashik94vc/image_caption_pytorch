"""
    Loads dataset into tensor for torch
"""
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from core.lib import loadData

class ImagenetDataset(Dataset):
    """
        Helper class to load dataset from h5 to create a data transform
    """
    def __init__(self, train, transform=None):
        self.dataset = loadData('dataset/imagenet.h5')
        self.train = train
        self.transform = transform
        self.class_names = self.dataset['class_index']
        if self.train:
            self.dataset = self.dataset['train']
        else:
            self.dataset = self.dataset['val']

    def __len__(self):
        return len(self.dataset['target'])

    def __getitem__(self, idx):
        sample_data = tuple()
        image_data = self.dataset['data'][idx]
        # image_data = image_data.transpose(1, 2, 0)
        # image_data = Image.fromarray(image_data)
        if self.transform:
            image_data = self.transform(image_data)
        sample_data = (image_data, self.dataset['target'][idx])

        return sample_data
