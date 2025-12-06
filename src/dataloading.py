import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
###############################################################
# 1. GLOBAL LABEL MAPPING
###############################################################

ALZ_MAP = {
    "Mild_Demented": 1,
    "Moderate_Demented": 1,
    "Very_Mild_Demented": 1,
    "Non_Demented": 0
}

TUMOR_MAP = {
    "Glioma": 2,
    "Meningioma": 2,
    "Pituitary": 2,
    "No Tumor": 0
}


###############################################################
# 2. COMBINED DATASET CLASS
###############################################################

class BrainDataset(Dataset):
    def __init__(self, root_dirs, label_maps, transform=None):
        """
        root_dirs: list of dataset dirs (e.g. ['data/alzheimer', 'data/tumor', ...])
        label_maps: list of dicts parallel to root_dirs
        """
        self.samples = []
        self.transform = transform

        for root, label_map in zip(root_dirs, label_maps):
            for class_name in os.listdir(root):
                class_path = os.path.join(root, class_name)
                if not os.path.isdir(class_path):
                    continue

                # map to global label
                global_label = label_map[class_name]

                # collect all images
                images = glob.glob(class_path + "/*")
                for img_path in images:
                    self.samples.append((img_path, global_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load image
        img = Image.open(img_path).convert("RGB")  # ensures 3-channel

        if self.transform:
            img = self.transform(img)

        return img, label


###############################################################
# 3. FUNCTION TO CREATE TRAIN / VAL LOADERS
###############################################################

def create_dataloaders(
    alz_path="data/alzheimer",
    tumor_a_path="data/tumor_a",
    tumor_b_path="data/tumor_b",
    img_size=224,
    batch_size=32,
    val_split=0.2,
    shuffle=True
):

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = BrainDataset(
        root_dirs=[alz_path, tumor_a_path, tumor_b_path],
        label_maps=[ALZ_MAP, TUMOR_MAP, TUMOR_MAP],
        transform=transform_train
    )

    # train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # apply validation transform
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
