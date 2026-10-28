"""Train the U-Net on AI4Mars Navcam (runs locally or on Colab)."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.metrics import confusion_matrix, iou_from_confusion
from src.dataset import build_pairs, split_pairs, MarsTerrainDataset
from src.unet import UNet 
from pathlib import Path 
import json 
import argparse

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
def save_checkpoint(path, model, optimizer, epoch, scheduler, best_miou):
    # writes the training state to the disk 
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), # Adam keeps its own running statistics for every weight
        "epoch": epoch,
        "scheduler": scheduler.state_dict(), # a resumed run won't restart the learning rate curve from the top
        "best_miou": best_miou,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


# the data paths differ, so read the settings from the command line 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()
    # on Colab you’d run: python -m src.train --data_dir /content/data/msl/ncam --out_dir /content/drive/MyDrive/mars_runs


# training, validation, per-class IoU, checkpoints, resume and history
# logs the curves so you can see training vs validation over time
def main(): # no args, creates everything itself and then runs the training 
    # 1. settings: data path, image size, batch size, learning rate, epochs, output folder
    # 2. device (GPU if available)
    # 3. build pairs → split → datasets → DataLoaders
    # 4. model, loss (ignore_index=255), optimizer, LR scheduler
    # 5. resume from a checkpoint if one exists
    # 6. loop over epochs: train → evaluate → log → save checkpoint (+ "best" if mIoU improved)
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # create the output folder 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data 
    data_dir = Path(args.data_dir)
    all_pairs = build_pairs(data_dir / "images" / "edr", data_dir / "labels" / "train")
    train_pairs, val_pairs = split_pairs(all_pairs)
    if args.limit: # with --limit 50 that gives 50 training and 10 validation images, randomly chosen 
        train_pairs = train_pairs[:args.limit]
        val_pairs = val_pairs[:args.limit//5]
    # wrap the pairs in the dataset 
    train_data = MarsTerrainDataset(train_pairs, size=args.size)
    val_data = MarsTerrainDataset(val_pairs, size=args.size)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # model, loss, optimizer, scheduler
    model = UNet().to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # not a fixed learning rate but scheduled
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # resume if a checkpoint exists
    start_epoch, best_miou, history = 0, 0.0, []
    last_path = out_dir / "last.pth"
    if last_path.exists():
        # loading 
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        scheduler.load_state_dict(ckpt["scheduler"])
        best_miou = ckpt["best_miou"]

        if (out_dir / "history.json").exists():
            with open(out_dir / "history.json", "r") as f:
                history = json.load(f) # to not lose the history 

    # epoch loop 
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, iou, miou = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        iou_str = [round(x, 3) for x in iou.tolist()] # its a tensor with 4 values, turn it into a list of rounded numbers
        print(f"Epoch number: {epoch}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, mIoU: {miou:.4f}, IoU: {iou_str}") # epoch, train loss, val loss, mIoU
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "miou": float(miou), # JSON cant store tensors 
            "iou": iou.tolist() # has 4 values, one per class
        })

        # save history to out_dir / "history.json" 
        # to later load it in a notebook and plot the training/validation curves 
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f) # overwriting the file each epoch 

        if miou > best_miou:
            best_miou = float(miou) 
            save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, scheduler, best_miou)
        save_checkpoint(last_path, model, optimizer, epoch, scheduler, best_miou) # always latest state 

if __name__ == "__main__":
    main()