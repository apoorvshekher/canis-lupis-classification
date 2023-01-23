import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from utils.train import train_model
from utils.test import test_model
from data.dataset import StanfordDogsDataset
from prepare_models.resnet50 import init_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = StanfordDogsDataset()

# Define loss function and optimizer
n_epochs = 1
loss_fn = nn.CrossEntropyLoss()

resnet50_model = init_resnet50(device)
optimizer = torch.optim.Adam(resnet50_model.parameters(), lr=0.01)
resnet50_model, history = train_model(train_dataset, train_loader, val_dataset, val_loader, resnet50_model, loss_fn, optimizer, n_epochs, device)
torch.save(resnet50_model, "app/models/model.pth")
test_model(test_dataset, test_loader, resnet50_model, device)