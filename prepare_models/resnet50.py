from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

def init_resnet50(device):

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.Linear(256, 120)
    )
    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model