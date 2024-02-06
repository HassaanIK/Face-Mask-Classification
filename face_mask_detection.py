
# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
# %matplotlib inline

"""#Initializing the Data"""

data_path = "face-mask-12k-images-dataset\Face Mask Dataset"
train_path = "face-mask-12k-images-dataset\Face Mask Dataset\Train"
test_path = "face-mask-12k-images-dataset\Face Mask Dataset\Test"
val_path = "face-mask-12k-images-dataset\Face Mask Dataset\Validation"

"""Trasform data into tensors and some other transformations too.."""

transforms = tt.Compose([
    tt.Resize((224,224)),
    tt.RandomHorizontalFlip(p=0.9),
    tt.ToTensor() ])

from torchvision.datasets import ImageFolder
from torchvision import transforms as tt


train_ds = ImageFolder(train_path,  transform=transforms)

val_ds = ImageFolder(val_path, transform=transforms)

test_ds = ImageFolder(test_path, transform=transforms)


batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)
test_dl = DataLoader(test_ds, batch_size)



"""Classes are equaly distributed in all the datasets...

#Model

Designing a base class having all the step functions of training
"""

class Base(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

        # print(f'Epoch: {epoch} | Train_loss: {result['train_loss']} | Val_loss:{result['val_loss']} | Val_acc: {result['val_acc']}')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_CHANNELS=3

import torch.nn as nn
import torch

class FaceMaskDetectionModel(Base):
    def __init__(self):
        super(FaceMaskDetectionModel, self).__init__()

        # Define the network layers
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn. Flatten()
        )

        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 28 * 28, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x):
        # Pass the input through the network
        x = self.network(x)

        # Reshape the output before passing it to the classifier
        x = x.view(x.size(0), -1)


        # Pass the output of the network through the classifier
        x = self.classifier(x)

        return x

model = FaceMaskDetectionModel()







@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False
)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

"""Training the model to get the best predictions and accuracy..."""

lr = 0.001
num_epochs = 5
opt_func = torch.optim.Adam

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)



# torch.save(model, 'facemask_detection_model.pth')

# torch.save(model.state_dict(), 'facemask_model_statedict1.pth')