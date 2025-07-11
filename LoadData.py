from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch
from collections import Counter

class AddNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Add Gaussian noise and clamp to [0, 1] for image tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0., 1.)

def transform(resize, rotation_degree=10, brightness=0.1, contrast=0.15, noise=False, noise_mean=0, noise_std=0.05, norm=True, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    transform_list = [
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degree),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
    ]
    transform_list.append(transforms.ToTensor())
    if noise:
        transform_list.append(AddNoise(noise_mean, noise_std))
    if norm:
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    transform = transforms.Compose(transform_list)
    return transform

def get_loader(batch_size=64, transform=transforms.ToTensor(), num_workers=0):
    torch.manual_seed(0)
    # load images dataset
    ImageFolder_Path = r'D:\PythonProject\COVID19\COVID-19_Radiography_Dataset\ImageDataset' # change path
    img_dataset = datasets.ImageFolder(root=ImageFolder_Path, transform=transform)

    # Split the dataset
    total_size = len(img_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    train_set, val_set, test_set = random_split(img_dataset, [train_size, val_size, test_size])

    # check numbers of pictures in each class -> COVID: 3616; Lung_Opacity: 6012; Normal: 10192
    train_labels = [img_dataset.samples[i][1] for i in train_set.indices]
    num_classes = len(img_dataset.classes)
    total_train = len(train_labels)
    class_counts = Counter(train_labels)
    weights = [total_train / (num_classes * class_counts[i]) for i in range(num_classes)]
    # assign a weight to each sample
    sample_weights = np.array([weights[label] for label in train_labels])
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers) # 310
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) # 31
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) # 31

    return train_loader, val_loader, test_loader, torch.tensor(weights)



