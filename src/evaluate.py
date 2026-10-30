"""Evaluate a trained U-Net checkpoint on the AI4Mars expert test sets."""

import torch
from torch.utils.data import DataLoader
from src.metrics import confusion_matrix, iou_from_confusion
from src.dataset import build_pairs, MarsTerrainDataset
from src.unet import UNet 
from pathlib import Path 
import json 
import argparse

# runs once, on the tests sets, after training and only needs the metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="results.json")
    return p.parse_args()


def load_model(checkpoint_path, device):
    model = UNet()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model 


# like evaluate() in train.py, but without the loss
def test_confusion(model, loader, device, num_classes=4):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad(): 
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            cm += confusion_matrix(preds.argmax(dim=1), lbls).cpu() 
    return cm


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    results = {}
    for name in ["min1", "min2", "min3"]:
        lbl_dir = data_dir / "labels" / "test" / f"masked-gold-{name}-100agree"
        pairs = build_pairs(data_dir / "images"  / "edr", lbl_dir)
        dataset = MarsTerrainDataset(pairs, size=args.size)
        loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=False)
        cm = test_confusion(model, loader, device)
        iou, miou = iou_from_confusion(cm)
        results[name] = {"len_pairs": len(pairs), 
                         "iou": iou.tolist(), 
                         "miou": float(miou), 
                         "confusion_matrix": cm.tolist(),
                        }
        # print the name, number of images, per-class IoU, mIoU
        print(f"Name: {name}, Length of pairs: {len(pairs)}, mIoU: {miou:.4f}, IoU: {[round(x, 3) for x in iou.tolist()]}")
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()