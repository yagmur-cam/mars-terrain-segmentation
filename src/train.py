''''
trains on all ~14,300 training images, on Colab’s GPU
evaluates on the validation set after every epoch (loss + mIoU)
saves checkpoints to Google Drive, so a Colab disconnect doesn’t lose your work
logs the curves so you can see training vs validation over time
'''

"""Train the U-Net on AI4Mars Navcam (runs locally or on Colab)."""

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # one pass over the training set → returns the average training loss
    pass


def evaulate(model, loader, loss_fn, device, num_classes=4):
    # one pass over the validation set, no gradients
    # → returns the average validation loss, per-class IoU, and mIoU
    pass



def save_checkpoint(path, model, optimizer, epoch, best_miou):
    # writes the training state to the disk 
    pass



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