import json
from pathlib import Path

import pandas as pd
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import helper

class OOCDataset(Dataset):
    """
    This is a class for the OOC images, which are stored in one single folder "datasets/ooc"
    In this dataset, the metadata file "datasets/ooc/dataset_metadata.json" is given.
    """
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

class OOCBackgroundOnly(Dataset):
    """
    This is a class for the background images alone in the OOC_Dataset, in which the objects are removed manually. \n
    The images are in the folder: dataset/OOC_Dataset/02_backgrounds/images_edited \n
    The metadata file is: dataset/OOC_Dataset/02_backgrounds/backgrounds_metadata.csv \n

    Arguments:\n
    ``root`` -- the dataset root (default: datasets/OOC_Dataset/02_backgrounds) \n
    ``transform`` -- transform function applied on images (default: None)
    """
    def __init__(self, root='datasets/OOC_Dataset/02_backgrounds', transform=None):
        data_path = Path(root) / "images_edited"
        metadata_path = next(Path(root).glob("*metadata.csv"))

        # save image paths
        self.image_paths = [path for path in sorted(Path(data_path).glob("*.JPEG"))]

        # get labels
        df = pd.read_csv(metadata_path)
        dataset_id_list = [path.stem for path in self.image_paths]
        class_hash_list =  [df[df['dataset_id'] == dataset_id].iloc[0]['class_hash'] for dataset_id in dataset_id_list]
        self.labels = helper.find_class_index_by_class_hash(class_hash_list)

        # transform
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label, image_path.stem

def load_dataset(data_path, batch_size, num_workers, transform, task):
    """
    load dataset using dataloader
    :param data_path: str | Path -- path to image folder
    :param batch_size: int -- batch size
    :param num_workers: int -- the number of workers
    :param transform: -- transformer function
    :param task: str -- dataset on which the inference is executed
    :return:
    """
    if task == "imagenet":
        print('Loading ImageNet2012 Validation Set...')
        dataset = ImageNet(data_path, split='val', transform=transform)
    elif task == "ooc":
        print('Loading Out-of-Context Dataset...')
        dataset = OOCDataset(root=data_path, transform=transform)
    elif task == "background":
        print("Loading Background Dataset...")
        dataset = OOCBackgroundOnly(root=data_path, transform=transform)
    elif task == "ranked": # TODO
        pass
    elif task == "placement": # TODO
        pass

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