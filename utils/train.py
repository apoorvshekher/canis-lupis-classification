import torch
from tqdm.auto import tqdm

def train_model(train_dataset, train_loader, val_dataset, val_loader, model, loss_fn, optimizer, n_epochs, device):

    history = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
        'val_loss' : []
    }

    for epoch in range(n_epochs):

        model.train()

        tr_acc = 0
        val_acc = 0
        running_loss = 0.0
        running_val_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_pred = torch.max(outputs, 1).indices
            tr_acc += int(torch.sum(train_pred == labels))

        history['train_acc'].append(tr_acc/len(train_dataset))
        history['train_loss'].append(running_loss/len(train_loader))

        model.eval()
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_val_loss += loss.item()

            val_pred = torch.max(outputs, 1).indices
            val_acc += int(torch.sum(val_pred == labels))

        history['val_acc'].append(val_acc/len(val_dataset))
        history['val_loss'].append(running_val_loss/len(val_loader))

        print(f"Epoch [{epoch+1}/{n_epochs}] | Train Acc : {history['train_acc'][-1]} | Train Loss : {history['train_loss'][-1]} | Val Acc : {history['val_acc'][-1]} | Val loss : {history['val_loss'][-1]}")
    
    return model, history