import torch
import random

class FedDataset(torch.utils.data.Dataset):
    def __init__(self, image_label_pairs, transform=None):
       
        self.image_label_pairs = image_label_pairs
        
        random.shuffle(self.image_label_pairs)

        self.transform = transform

    def __len__(self):
        
        return len(self.image_label_pairs)

    def __getitem__(self, idx):

        image, label = self.image_label_pairs[idx]               

        if self.transform:
            image = self.transform(image)

        return image, label