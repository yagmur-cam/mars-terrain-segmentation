"""The confusion matrix and IoU functions"""

import torch 

def confusion_matrix(preds, labels, num_classes=4, ignore_index=255):
    # preds and labels are shape [B, H, W]
    # count every (true class, predicted class) combination into a 4×4 table
    # for each kept pixel, compute true * num_classes + pred, a number from 0 to 15, one per combination
    mask = labels != ignore_index
    # indexing a tensor with a T/F mask keeps only the pixels where the mask is, also flattens into 1D list 
    true = labels[mask]
    pred = preds[mask]
    index = true * num_classes + pred 
    count = torch.bincount(index, minlength=num_classes**2) # all 16 slots exist even if some combinations never occur
    return count.reshape(num_classes, num_classes) # turns the 16 counts into the 4×4 table


def iou_from_confusion(cm): # works on all 4 classes at once (vectorising)
    # TP (true-positive) for each class, the diagonal of the matrix
    tp = cm.diag()
    # all pixels predicted as class c = the column sum
    predicted = cm.sum(dim=0)
    # all pixels truly class c = the row sum 
    true = cm.sum(dim=1)
    # IoU = TP / (predicted + true − TP)
    iou = tp / (predicted +  true - tp)
    return iou, torch.nanmean(iou) # a class with no data at all (0/0 = nan) is left out of the average


if __name__ == "__main__":
    # test 
    labels = torch.tensor([[0, 0, 1, 255]])
    preds = torch.tensor([[0, 1, 1, 2]])
    cm = confusion_matrix(preds, labels)
    print(cm) 
    print(iou_from_confusion(cm)) # tensor([0.5, 0.5, nan, nan]), tensor(0.5)