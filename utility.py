import os
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import optuna
import tfrecord
import torch
from torch.utils.data import Dataset


# From https://www.kaggle.com/code/adikaboost/transfer-learning-efficientnet-pytorch
def transform_tf_to_df(dataset_path, subset_data):
    df = pd.DataFrame({"id": pd.Series(dtype="str"), 
                       "class": pd.Series(dtype="int"), 
                       "img": pd.Series(dtype="object")})    
    tf_files = []
    
    for subdir, dirs, files in os.walk(dataset_path):
        if subdir.split("/")[-1] == subset_data:
            for file in files:
                filepath = subdir + os.sep + file
                tf_files.append(filepath)
                
    for tf_file in tf_files:
        if subset_data == "test":
            loader = tfrecord.tfrecord_loader(tf_file, None, {"id": "byte", "image": "byte"})
        else:
            loader = tfrecord.tfrecord_loader(tf_file, None, {"id": "byte","image": "byte", "class": "int"})
        
        for record in loader:
            id_label = record["id"].decode('utf-8')
            label = record["class"][0].item() if subset_data != "test" else None
            img_bytes = np.frombuffer(record["image"], dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            df.loc[len(df.index)] = [id_label, label, img]
    return df

class FlowerDataset(Dataset):
    def __init__(self, dataset_path, subset_data, num_classes=104, transform=None):
        self.num_classes = num_classes
        self.transform = transform
        self.df_data = transform_tf_to_df(dataset_path, subset_data)

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        "Iterable function which applies to each row"
        img_id = self.df_data.iloc[idx, 0]
        label = self.df_data.iloc[idx, 1]
        image = self.df_data.iloc[idx, 2]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        y = np.zeros(self.num_classes, dtype=np.float32)
        y[label] = int(1)
        return img_id, y, image

def model_trainer(model, criterion, optimizer, train_loader, val_loader=None, epochs=0, scheduler=None, device='cpu', show_progress=False, trial=None):
    # Move the model to the specified device
    model.to(device)
    
    history = {
        'train_loss': [],
        'train_accuracy': []
    }
    
    if val_loader is not None:
        history['val_loss'] = []
        history['val_accuracy'] = []
    
    epochs_loop = tqdm(range(0, epochs), desc="Epoch", leave=True) if show_progress else range(0, epochs)
    for epoch in epochs_loop:        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # training loop
        training_loop = tqdm(train_loader, leave=False, desc="Training") if show_progress else train_loader
        for _, labels, inputs in training_loop:
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
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            _, y_class = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted == y_class).sum().item()

            if show_progress:
                training_loop.set_postfix(training_loss=running_loss / (training_loop.n + 1), training_accuracy=correct / total)
            
        # Calculate average loss and accuracy for the epoch
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_accuracy = correct / total

        if show_progress:
            epochs_loop.set_postfix(train_loss=train_epoch_loss, train_accuracy=train_epoch_accuracy)
            
        history['train_loss'].append(train_epoch_loss)
        history['train_accuracy'].append(train_epoch_accuracy)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            val_loop = tqdm(val_loader, leave=False, desc="Validating") if show_progress else val_loader
            
            with torch.no_grad():
                for _, labels, inputs in val_loop:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, y_class = torch.max(labels.data, 1)
                    
                    val_total += labels.size(0)
                    val_correct += (predicted == y_class).sum().item()

                    if show_progress:
                        val_loop.set_postfix(val_loss=val_loss / (val_loop.n + 1), val_accuracy=val_correct / val_total)
            
            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_accuracy = val_correct / val_total

            if show_progress:
                epochs_loop.set_postfix(train_loss=train_epoch_loss, val_loss=val_epoch_loss, train_accuracy=train_epoch_accuracy,  val_accuracy=val_epoch_accuracy)
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_accuracy)

        if scheduler is not None:
            scheduler.step()

        if trial is not None:
            rep_acc = val_epoch_accuracy if val_loader is not None else train_epoch_accuracy
            trial.report(rep_acc, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return history