from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor

def StanfordDogsDataset():
    train_dataset = ImageFolder('data/images/train', transform=Compose([Resize((299,299)),ToTensor()]))
    val_dataset = ImageFolder('data/images/val', transform=Compose([Resize((299,299)),ToTensor()]))
    test_dataset = ImageFolder('data/images/test', transform=Compose([Resize((299,299)),ToTensor()]))

    train_loader = DataLoader(
        train_dataset,
        batch_size = 32
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size = 32
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = 32
        )
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader