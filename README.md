# Mars Terrain Segmentation with the AI4Mars Dataset 

## Overview
what the model does (pixel-wise terrain classes from Curiosity Navcam images), why it matters (rover path planning), and “U-Net built from scratch in PyTorch”.

## Results 

| Run | Change | min1 mIoU | min2 mIoU | min3 mIoU | Big rock IoU (min1) |
|-----|--------|-----------|-----------|-----------|---------------------|
| run1 | baseline, 256px | 0.650 | 0.697 | 0.751 | 0.079 |
| run2 | class-weighted ross-entropy | 0.655 | 0.696 | 0.764 | 0.088 |
| run3 |           | | | | |

**Note:** min1/min2/min3: expert test labels where at least 1, 2 or 3 annotators agreed.

- the main classes score 0.92–0.97 IoU on min3

- stricter test sets score higher (min3 > min2 > min1) because the model fails mainly where experts disagree

- big rock is the weak class (0.08 on min1; the min3 number is based on only 5 images)

```
recall:     run1 0.315  run2 0.606
precision:  run1 0.095  run2 0.093
bedrock→BR: run1 0.015  run2 0.032
soil→BR:    run1 0.000  run2 0.001
```

- run2: recall 0.32 → 0.61, precision ~0.09 in both runs, so IoU stays flat; weighting traded misses for false alarms on bedrock
- for a rover, missing a rock is worse than a false alarm, so run2 may be the safer model



    - One example prediction image (`example_1.png`).



## How it works 
data (16k crowd-labelled train images, expert test sets, unlabelled pixels ignored), model (U-Net, 31M parameters), training (CE loss, Adam, cosine schedule, AMP, 256px), evaluation (confusion matrix → IoU; why not pixel accuracy), and the overfit sanity check.

## Project structure

### How to run it 

tree + download the data (Zenodo link)

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the data:
```bash
```

Train:
```bash

```

Evaluate:
```bash

```

## Limitations 

greyscale Navcam only, crowd vs expert label mismatch on big rock, and shadows/dark scenes (point to notebook 06).

- 
- 
- 

## References 

add a link to each, and the dataset licence line (CC BY 4.0).

- AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars
- U-Net: Convolutional Networks for Biomedical Image Segmentation