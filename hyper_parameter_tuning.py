#!/usr/bin/env python
# coding: utf-8
import sys
import logging
import gc

import optuna

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import densenet161, efficientnet_v2_l, vgg19_bn
from torch.optim import Adam

# side file containing all the utility function and class
from utility import FlowerDataset, model_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
    logging.FileHandler("hyper_parameter_tuning.log"),
    logging.StreamHandler(sys.stdout)
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f"Device: {DEVICE}")

DATASET_PATH = "./tfrecords-jpeg-512x512"
# batch size is limited by the amount of available GPU memory (24Gb) 
BATCH_SIZE = 8
NUM_CLASSES = 104
IMG_RESOLUTION = (512, 512)

class RandomImgDropout(object):
    """
        Apply randomly drops out rectangular regions of an image by setting them to zero. 

        Attributes:
        - p (float): Probability of applying the dropout transformation. Default is 0.5.
        - dim (int): Dimension of the image (assuming a square image). Default is IMG_RESOLUTION[0].
        - n_dropout (int): Number of dropout regions to create in the image. Default is 5.
        - scaled_size (float): Size of the dropout regions as a fraction of the image dimension. Default is 0.1.

        Parameters:
            img (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Output image tensor of shape (C, H, W).
    """
    
    def __init__(self, p=0.5, dim=IMG_RESOLUTION[0], n_dropout=5, scaled_size=0.1):
        self.p = p
        self.dim = dim
        self.n_dropout = n_dropout
        self.scaled_size = scaled_size
            
    def __call__(self, img):
        do_tr = torch.rand(1)[0] < self.p
        
        if not do_tr:
            return img

        for _ in range(0, self.n_dropout):
            x = torch.randint(0, self.dim, ()).type(torch.int32)
            y = torch.randint(0, self.dim, ()).type(torch.int32)
            width = torch.tensor(self.scaled_size * self.dim, dtype=torch.int32)

            ya = torch.maximum(y-width//2, torch.tensor(0, dtype=torch.int32))
            yb = torch.minimum(y+width//2, torch.tensor(self.dim, dtype=torch.int32))
            xa = torch.maximum(x-width//2, torch.tensor(0, dtype=torch.int32))
            xb = torch.minimum(x+width//2, torch.tensor(self.dim, dtype=torch.int32))

            img[:, ya:yb, xa:xb] = 0
            
        return img

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=IMG_RESOLUTION, scale=(0.8, 1)),
    transforms.RandomEqualize(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ElasticTransform(alpha=80.0)]),
    transforms.RandomPerspective(distortion_scale=(0.3), p=0.4),
    transforms.PILToTensor(),
    RandomImgDropout(scaled_size=0.12, n_dropout=10),
    transforms.ToDtype(torch.float32),
    transforms.Normalize(*stats,inplace=True)
])

val_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ToDtype(torch.float32),
    transforms.Normalize(*stats,inplace=True)
])

logging.info("Loading datasets...")
train_data = FlowerDataset(DATASET_PATH, 'train', num_classes=NUM_CLASSES, transform=train_transform)
val_data = FlowerDataset(DATASET_PATH, 'val', num_classes=NUM_CLASSES, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True)


# The cell below have been converted to raw format to avoid accidentaly running the benchmark at each restart of the notebook.
models_benchmark = {}

epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()

def get_custom_classifier(linear_layers: list, dropout_layers: list, bn_layers: list):
    layers = []
    for out_features, dropout_rate, is_bn in zip(linear_layers, dropout_layers, bn_layers):
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.LazyLinear(out_features))
        if is_bn : layers.append(nn.LazyBatchNorm1d())
        layers.append(nn.ReLU())
        
    layers.append(nn.LazyLinear(NUM_CLASSES))
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

def objective(trial):
    logging.info(f"Starting Trial {trial.number}")
    
    model = efficientnet_v2_l(weights='DEFAULT')
    
    n_hidden_layers = trial.suggest_int('num_hidden_layers', 0, 3)
    linear_layers = []
    dropout_layers = []
    bn_layers = []
    
    for i in range(0, n_hidden_layers):
        linear_layers.append(trial.suggest_int(f'l{i}_out_features', 128, 1024, step=32))
        dropout_layers.append(trial.suggest_float(f'l{i}_dropout_rate', 0, 0.6))
        bn_layers.append(trial.suggest_categorical(f'l{i}_is_bn', [True, False]))

    model.classifier = get_custom_classifier(linear_layers, dropout_layers, bn_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    hist = model_trainer(model, criterion, optimizer, train_loader, val_loader=val_loader, epochs=15, device=DEVICE, show_progress=False, trial=trial)

    model.cpu()
    del model, criterion, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return hist['val_accuracy'][-1]


logging.info("Starting HPO...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, gc_after_trial=True, show_progress_bar=False)

logging.info("Results saved to HPO.csv")
HPO_df = study.trials_dataframe()
HPO_df.to_csv('./HPO.csv', index=False)