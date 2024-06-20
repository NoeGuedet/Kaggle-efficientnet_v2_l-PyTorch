import sys
import logging
import gc

import pandas as pd

import torch
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

from torchvision.models import efficientnet_v2_l

from sklearn.model_selection import KFold

# side file containing all the utility function and class
from utility import FlowerDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
    logging.FileHandler("k-fold_training.log"),
    logging.StreamHandler(sys.stdout)
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f'Device : {DEVICE}')

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
        - dim (int): Dimension of the image (assuming a square image).
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

N_WORKERS = 16

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

hidden_layers = [1024] 
dropout_layers = [0.38] 
bn_layers = [False]

# since the model is pretrained and the batch size is small, a small lr is better
head_lr_start = 1e-5
head_lr_min = 1e-5
head_lr_max = 1e-4
head_lr_rampup_epochs = 5
head_lr_sustain_epoch = 0
head_lr_decay = .8

def custom_head_lr_scheduler(epoch):
    if epoch < head_lr_rampup_epochs:
        return (head_lr_max - head_lr_start) / head_lr_rampup_epochs * epoch + head_lr_start
        
    elif epoch < head_lr_rampup_epochs + head_lr_sustain_epoch:
        return head_lr_max
        
    else:
        return (head_lr_max - head_lr_min) * head_lr_decay**(epoch - head_lr_rampup_epochs - head_lr_sustain_epoch) + head_lr_min

clr_lr_max = 1e-4
clr_lr_min = 1e-6
clr_lr_decay = 0.8

def custom_clr_lr_scheduler(epoch):
    lr = (clr_lr_max - clr_lr_min) * clr_lr_decay**(epoch) + clr_lr_min
    return lr

# Early stop class based of the trend of a given loss
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def es_training(model, optimizer, criterion, dataset, train_loader, val_loader, epochs, patience, min_delta, scheluder=None, device='cpu'):
    early_stopper = EarlyStopper(patience, min_delta)
    model.to(device)
    # Dictionary to store training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(0, epochs):        
        model.train()
        dataset.transform = train_transform
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for _, labels, inputs in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Calculate statistics            
            _, predicted = torch.max(outputs.data, 1)
            _, y_class = torch.max(labels.data, 1)
            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (predicted == y_class).sum().item()

        # Update lr
        if scheduler is not None:
            scheduler.step()
            
        # Calculate average training loss and accuracy for the epoch
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_accuracy = train_correct / train_total

        model.eval()
        dataset.transform = val_transform
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for _, labels, inputs in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Calculate statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, y_class = torch.max(labels.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == y_class).sum().item()

        # Calculate average validation loss and accuracy for the epoch
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = val_correct / val_total

        # update history
        history['train_loss'].append(train_epoch_loss)
        history['train_accuracy'].append(train_epoch_accuracy)
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_accuracy)

        logging.info(
            f'Epoch {epoch} completed | ' + 
            f'training loss = {train_epoch_loss} | ' +
            f'training accuracy = {train_epoch_accuracy} | ' +
            f'val loss = {val_epoch_loss} | ' +
            f'val accuracy = {val_epoch_accuracy}'
        )

        # check for early stop
        early_stopper.step(val_epoch_loss)
        if early_stopper.early_stop:
            logging.info('-'*5 + f'Early stop trigger --> stopping training' + '-'*5)
            break

    return history

def get_model():
    model = efficientnet_v2_l(weights='DEFAULT')
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    model.classifier = get_custom_classifier(hidden_layers, dropout_layers, bn_layers)
        
    return model

random_state = 3210
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

k_models = [get_model() for _ in range(0, k_folds)]

logging.info('Loading dataset ...')
dataset = FlowerDataset(DATASET_PATH, 'train+val')

# Early stop variables
es_patience = 5
es_min_delta = 0.01

criterion = nn.CrossEntropyLoss()
epochs = 50
histories = {}

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    logging.info(f'###### FOLD {fold} ######')
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=val_subsampler)

    model = k_models[fold]
    optimizer = Adam([
                        {'params': model.features.parameters()},
                        {'params': model.classifier.parameters()}
                    ], lr=1)
    
    scheduler = LambdaLR(optimizer, lr_lambda=[custom_head_lr_scheduler, custom_clr_lr_scheduler])
    
    history = es_training(
                    model, 
                    optimizer, 
                    criterion,
                    dataset,
                    train_loader, 
                    val_loader, 
                    epochs, 
                    patience=es_patience, 
                    min_delta=es_min_delta, 
                    scheluder = scheduler, 
                    device=DEVICE
                )
    
    histories[f'model_f-{fold}_train_accuracy'] = history['train_accuracy']
    histories[f'model_f-{fold}_val_accuracy'] = history['val_accuracy']
    histories[f'model_f-{fold}_train_loss'] = history['train_loss']
    histories[f'model_f-{fold}_val_loss'] = history['val_loss']
    
    save_path = f'./models/model_f-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # free GPU memory
    model.cpu()
    del scheduler, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

# Convert dict to dataframe with all the key's value not having the same size :
# https://stackoverflow.com/questions/38446457/filling-dict-with-na-values-to-allow-conversion-to-pandas-dataframe
logging.info('Saving result ...')
histories = pd.DataFrame.from_dict(histories, orient='index').T
histories.to_csv('k-fold_training_histories.csv', index=False)
logging.info('Training finished !')