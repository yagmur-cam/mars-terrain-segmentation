"""Train the U-Net on AI4Mars Navcam (runs locally or on Colab)."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.metrics import confusion_matrix, iou_from_confusion

# trains on ~14,300 training images, on Colab’s GPU
#the overfitting loop, generalized to go through all the batches of a DataLoader 
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # one pass over the training set → returns the average training loss
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, lbls)
        # clear old gradients → compute new ones → step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) # average loss per batch

# evaluate on the validation set after every epoch (loss + mIoU)
# returns three things: the average validation loss, the per-class IoU, and the mIoU
def evaluate(model, loader, loss_fn, device, num_classes=4):
    # one pass over the validation set, no gradients
    # → returns the average validation loss, per-class IoU, and mIoU
    model.eval()
    total_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad(): # no training happens, no gradient 
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            total_loss += loss.item()
            # so that this batch's predictions and labels are a 4x4 table first and cm lives on the CPU 
            cm += confusion_matrix(preds.argmax(dim=1), lbls).cpu() 
    iou, miou = iou_from_confusion(cm)
    return (total_loss / len(loader)), iou, miou


# saves checkpoints to Google Drive, so a Colab disconnect doesn’t lose your work
def save_checkpoint(path, model, optimizer, epoch, best_miou):
    # writes the training state to the disk 
    pass



# logs the curves so you can see training vs validation over time
def main():
    # 1. settings: data path, image size, batch size, learning rate, epochs, output folder
    # 2. device (GPU if available)
    # 3. build pairs → split → datasets → DataLoaders
    # 4. model, loss (ignore_index=255), optimizer, LR scheduler
    # 5. resume from a checkpoint if one exists
    # 6. loop over epochs: train → evaluate → log → save checkpoint (+ "best" if mIoU improved)
    pass



if __name__ == "__main__":
    main()