import torch

def test_model(test_dataset, test_loader, model, device):

    model.eval()
    test_acc = 0
    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        test_pred = torch.max(outputs, 1).indices
        test_acc += int(torch.sum(test_pred == labels))

    print(f"Test Accuracy : {test_acc/ len(test_dataset)}")