# AI4Mars Navcam dataset: builds image/label pairs and loads them as PyTorch tensors

import numpy as np 
import torch
from torch.utils.data import Dataset
from PIL import Image 

class MarsTerrainDataset(Dataset):
    def __init__(self, pairs, size=512): # receives the list of (image path, label path) pairs and the target size (512), and stores them
        self.pairs = pairs
        self.size = size 

    def __len__(self): # returns how many pairs there are 
        return len(self.pairs)

    # The image should become a tensor of shape [1, 512, 512] with float values, channels × height × width
    # The label should become shape [512, 512] with integer values (torch.long), not float
    # Each label pixel is a class number (0–3, or 255)
    # scale the image, divide by 255, so values go from 0.0-1.0
    def __getitem__(self, i): # opens pair number i, resizes the image (bilinear) and the label (nearest), converts both to tensors, and returns them
        img_path, lbl_path = self.pairs[i]
        with Image.open(img_path) as im:
            im_resized = im.resize((self.size, self.size), Image.BILINEAR)
        im_arr = np.array(im_resized)
        image = torch.from_numpy(im_arr.astype(np.float32) / 255).unsqueeze(0)

        with Image.open(lbl_path) as lb:
            lb_resized = lb.resize((self.size, self.size), Image.NEAREST)
        lb_array = np.array(lb_resized)
        label = torch.from_numpy(lb_array).long()

        return image, label

# returns a list of (image_path, label_path) tuples
def build_pairs(img_dir, lbl_dir): # builds the pair list, matching names 
    img_by_stem = {img.stem: img for img in img_dir.glob("*.JPG")} # for each .JPG image, stem → path
    pairs = []

    for lbl in sorted(lbl_dir.glob("*.png")):
        stem = lbl.stem.replace("_merged", "")
        if stem not in img_by_stem:
            continue 
        with Image.open(lbl) as lb:
            if (np.array(lb) == 255).all():
                continue
        pairs.append((img_by_stem[stem], lbl))

    return pairs 

