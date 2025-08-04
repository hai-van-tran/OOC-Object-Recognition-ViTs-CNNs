import json
from pathlib import Path

import pandas as pd
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image


class OOCDataset(Dataset):
    def __init__(self, root='datasets/ooc', transform=None):
        # save image paths
        self.image_paths = [path for path in sorted(Path(root).glob('*.png'))]

        # get labels
        metadata_path = Path(root) / 'dataset_metadata.json'
        with open(metadata_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            self.labels = [df[df['filename'] == filename.name].iloc[0]['object_class_id']
                     for filename in self.image_paths]

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
           image = self.transform(image)
        label = self.labels[idx]
        return image, label, image_path.name


def load_dataset(data_path, batch_size, num_workers, transform, ooc_dataset):
    if ooc_dataset:
        print('Loading Out-of-Context Dataset...')
        dataset = OOCDataset(root=data_path, transform=transform)
    else:
        print('Loading ImageNet2012 Validation Set...')
        dataset = ImageNet(data_path, split='val', transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    progress_bar = tqdm(dataloader)

    return progress_bar

if __name__ == '__main__':
    pass