import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['id_code'][idx]
        image_path = os.path.join(self.data_dir, f"{image_id}.png")
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.data['diagnosis'][idx]
        
        return image, label
