## Environment Setup

This project has been tested with the following environment:

| Component | Version | Check Command |
|------------|----------|----------------|
| **OS** | Ubuntu 22.04 LTS | `lsb_release -a` |
| **Python** | 3.10.x | `python3 --version` |
| **CUDA Compiler** | 11.5 | `nvcc --version` |
| **NVIDIA Driver** | 580.95.05 | `nvidia-smi` |
| **PyTorch** | 2.1.2 (CUDA enabled) | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |

---

### ⚙️ Quick Installation

Copy and paste the following commands into your terminal:

```bash
# Clone repository
git clone https://github.com/AAU-Underwater-Robot-Lab/mmsegmentation
cd mmsegmentation

# Create and activate virtual environment
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-venv git
python3 -m venv .venv
source .venv/bin/activate

# Dont change orders of dependencies. 
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
pip install albumentations
# ensure the last thing before pip install -v -e . is nump<2.
# Several packages will install lates version of numpy, but for some reason mmengine expect a numpy version 1.X
pip install "numpy<2"
# Install this repository in editable mode
pip install -v -e .
pip install ftfy
pip install regex
pip install tensorboard

```

### Verify Setup

Varify that torch can access cuda-compiler and GPU.
```bash
# Verify setup
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"
```
Expected outcome: Torch: 2.1.2+cu121 | CUDA: 12.1 | GPU: Tesla T4

Verify that you can download pre-trained data and make a inference using MMEngine.

```bash
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```
Check on project directory. result.jpg is generated

# Training on Custom Dataset

## Setting up Custom Data

### Dataset Structure

MMSegmentation expects datasets to follow a **COCO-like** or **custom folder-based structure** where each split (`train`, `val`, `test`) has its own image and annotation subdirectories.

A typical structure looks like this:

```
data/
├── my_dataset/
│ ├── train/
│ │ ├── images/
│ │ │ ├── 0001.png
│ │ │ ├── 0002.png
│ │ │ └── ...
│ │ └── masks/
│ │ ├── 0001.png
│ │ ├── 0002.png
│ │ └── ...
│ ├── val/
│ │ ├── images/
│ │ └── masks/
│ └── test/
│ ├── images/
│ └── masks/
```

Each **mask image** should:
- Have the same filename as its corresponding RGB image.
- Use **grayscale pixel values** where each integer represents a class label (e.g., `0 = background`, `1 = object1`, `2 = object2`, ...).
- Be stored in `.png` format (recommended).

### Add Custom Dataset Metadata

To make MMSegmentation recognize a **new dataset**, you must register it by creating a small Python class that defines its **metadata** — such as class names, color palette, and file suffixes.

This file tells MMSegmentation **how to interpret your dataset’s images and masks**.

---

Place this file under: mmsegmentation/mmseg/datasets/

Example file python file `underwater_turbidity_aau.py`

```python

# ============================================================
# Custom Dataset Definition for MMSegmentation
# ============================================================
# This file defines how MMSegmentation should load, interpret,
# and visualize your dataset (e.g., underwater segmentation).
#
# Location:
#   mmsegmentation/mmseg/datasets/underwater_turbidity_aau.py
# ============================================================

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class UnderwaterTurbidityAAU(BaseSegDataset):
    """Custom underwater dataset using grayscale masks.
    
    Each pixel in the mask corresponds to a class ID (0–9).
    Classes and colors are defined in the METAINFO dictionary.
    """

    # ------------------------------------------------------------
    # METAINFO: dataset metadata (class names, colors, etc.)
    # ------------------------------------------------------------
    METAINFO = dict(
        # Define the semantic classes your dataset contains
        classes=(
            'Pool_background',
            'Black_Pipe',
            'Aluminum_Pipe',
            'Wooden_piece',
            'Granite',
            'Acomar',
            'Can',
            'Benchy',
            'Intruder',
            'BlueROV2'
        ),

        # Color palette used for visualization (RGB format)
        palette=[
            [160, 196, 189],  # Pool_background
            [141, 195, 141],  # Black_Pipe
            [199, 187, 116],  # Aluminum_Pipe
            [164,  92, 174],  # Wooden_piece
            [197, 197, 197],  # Granite
            [ 58, 105, 176],  # Acomar
            [179,  63,  62],  # Can
            [202, 160, 133],  # Benchy
            [207, 106, 189],  # Intruder
            [150, 151, 170],  # BlueROV2
        ],

        # (OPTIONAL) You can also include additional metadata keys here:
        #
        # 'dataset_name': 'UnderWaterAAU',
        # 'description': 'Annotated underwater scenes for segmentation',
        # 'label_map': {0: 'background', 1: 'pipe', ...},  # if you remap labels
        # 'ignore_index': 255,  # value to ignore during training
    )

    # ------------------------------------------------------------
    # Dataset Initialization
    # ------------------------------------------------------------
    def __init__(self,
                 img_suffix='.png',        # File extension for input images
                 seg_map_suffix='.png',    # File extension for segmentation masks
                 reduce_zero_label=False,  # Ignore label 0 (background) if True
                 **kwargs):
        """Initialize the dataset and pass extra arguments to BaseSegDataset.

        Optional kwargs examples:
          - data_root: root directory (e.g. 'data/underwater_3/')
          - pipeline: augmentation/transformation pipeline
          - ann_file: path to annotation list (if using COCO format)
        """

        # Initialize parent class (BaseSegDataset handles image/mask pairing)
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
```

Next open `mmsegmentation/mmseg/datasets/__init__.py` and add the created Class in **__all__** list.

```python
...
from .underwater_turbidity_aau import UnderwaterTurbidityAAU
...
__all__ = [
    'BaseSegDataset', 'BioMedical3DRandomCrop', 'BioMedical3DRandomFlip',
    'CityscapesDataset', 'PascalVOCDataset', 'ADE20KDataset',
    'PascalContextDataset', 'PascalContextDataset59', 'ChaseDB1Dataset',
    'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'DarkZurichDataset',
    'NightDrivingDataset', 'COCOStuffDataset', 'LoveDADataset',
    'MultiImageMixDataset', 'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset',
    'LoadAnnotations', 'RandomCrop', 'SegRescale', 'PhotoMetricDistortion',
    'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'RandomCutOut', 'RandomMosaic', 'PackSegInputs', 'ResizeToMultiple',
    'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'DecathlonDataset', 'LIPDataset', 'ResizeShortestEdge',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedicalRandomGamma', 'BioMedical3DPad', 'RandomRotFlip',
    'SynapseDataset', 'REFUGEDataset', 'MapillaryDataset_v1',
    'MapillaryDataset_v2', 'Albu', 'LEVIRCDDataset',
    'LoadMultipleRSImageFromFile', 'LoadSingleRSImageFromFile',
    'ConcatCDInput', 'BaseCDDataset', 'DSDLSegDataset', 'BDD100KDataset',
    'NYUDataset', 'HSIDrive20Dataset', 'UnderWaterTurbidityAAU'
]
```
### Dataloaders in MMSegmentation

In MMSegmentation, **dataloaders** define how images and annotations are loaded, transformed, batched, and fed into the model during training and evaluation.  

They are configured inside the dataset config file — typically under: `mmsegmentation/configs/_base_/datasets/`

Each dataloader includes three main parts:
- **Dataset definition** (where to find data and what format to use)
- **Pipeline** (how to preprocess and augment the data)
- **Sampler + loader settings** (batch size, number of workers, etc.)

---

Example: Underwater Dataset Configuration

```python
# ============================================================
# Underwater Dataset Configuration (BaseSegDataset)
# ============================================================
# This configuration defines how MMSegmentation loads and processes
# your underwater dataset for training, validation, and testing.
#
# It includes:
# 1. Dataset metadata (type, root directory)
# 2. Data pipelines (augmentation & preprocessing)
# 3. Dataloaders (how data is batched and sampled)
# 4. Evaluators (metrics for validation/testing)
# ============================================================

# ------------------------------------------------------------
# Dataset Type and Root Path
# ------------------------------------------------------------
# 'dataset_type' must match the name of your registered dataset class.
# This links to the dataset defined in mmseg/datasets/underwater_aau.py
dataset_type = 'UnderwaterTurbidityAAU'

# Root directory where images and annotations are stored.
data_root = 'data/underwater_3/'


# ------------------------------------------------------------
# Data Pipelines
# ------------------------------------------------------------
# The pipeline defines how data is loaded, augmented, and prepared
# before being passed into the model. Training and test pipelines
# differ slightly (e.g., augmentations vs. deterministic resizing).

crop_size = (512, 512)  # Random crop size for training images

# ---------------------------
# Training Pipeline
# ---------------------------
train_pipeline = [
    # Step 1: Load RGB image
    dict(type='LoadImageFromFile'),

    # Step 2: Load segmentation mask (grayscale image)
    dict(type='LoadAnnotations', reduce_zero_label=False),

    # Step 3: Randomly resize image within a given ratio range
    dict(
        type='RandomResize',
        scale=(1920, 1080),       # target scale
        ratio_range=(0.5, 2.0),  # random scaling factor
        keep_ratio=True),        # maintain aspect ratio

    # Step 4: Randomly crop a patch from the image
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    # Step 5: Randomly flip the image horizontally
    dict(type='RandomFlip', prob=0.5),

    # Step 6: Apply random photometric distortions
    dict(type='PhotoMetricDistortion'),

    # Step 7: Pack results into model input format (tensors)
    dict(type='PackSegInputs')
]

# ---------------------------
# Test / Validation Pipeline
# ---------------------------
test_pipeline = [
    # Step 1: Load image
    dict(type='LoadImageFromFile'),

    # Step 2: Resize for evaluation
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),

    # Step 3: Load annotations (after resize)
    # Ground-truth masks don’t need random transformations
    dict(type='LoadAnnotations', reduce_zero_label=False),

    # Step 4: Pack into model input
    dict(type='PackSegInputs')
]


# ------------------------------------------------------------
# Dataloaders
# ------------------------------------------------------------
# These control how data is read, batched, and fed into the model.
# Each dataloader specifies its dataset, sampler, batch size,
# number of workers, and preprocessing pipeline.

# ---------------------------
# Training Dataloader
# ---------------------------
train_dataloader = dict(
    batch_size=8,                # number of samples per GPU
    num_workers=4,               # number of subprocesses for loading data
    persistent_workers=True,     # keep workers alive between epochs
    sampler=dict(
        type='InfiniteSampler',  # loops through dataset infinitely
        shuffle=True             # shuffle data order each epoch
    ),
    dataset=dict(
        type=dataset_type,       # use our custom dataset class
        data_root=data_root,     # path to dataset root
        img_suffix='.png',       # image file extension
        seg_map_suffix='.png',   # mask file extension
        data_prefix=dict(        # subfolder definitions
            img_path='images/train',             # where to find training images
            seg_map_path='annotations/train_mapped'  # where to find masks
        ),
        pipeline=train_pipeline,  # apply augmentation pipeline
    ),
)

# ---------------------------
# Validation Dataloader
# ---------------------------
val_dataloader = dict(
    batch_size=1,                # evaluate one image per step
    num_workers=4,
    persistent_workers=False,    # workers can shut down between epochs
    sampler=dict(
        type='DefaultSampler',   # iterate dataset once per epoch
        shuffle=False            # do not shuffle validation data
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val_mapped'
        ),
        pipeline=test_pipeline,   # deterministic preprocessing
    ),
)

# Use the same dataloader setup for testing
test_dataloader = val_dataloader


# ------------------------------------------------------------
# Evaluators (Metrics)
# ------------------------------------------------------------
# Evaluators define which metrics to compute after each validation/test run.
# 'mIoU' (Mean Intersection over Union) and 'mDice' (Mean Dice Coefficient)
# are standard segmentation metrics for performance evaluation.

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice']
)

# Test evaluator mirrors validation setup
test_evaluator = val_evaluator
```

### Final step: training on custom data.

Once you have registered your custom dataset (see `UnderWaterTurbidityAAU` in `mmseg/datasets/`), you can train any MMSegmentation model (like **DeepLabV3**) on it by creating a **custom configuration file**.

This file defines:
- The model architecture (e.g., DeepLabV3 with ResNet-50 backbone)
- The dataset configuration (train/val/test)
- Runtime and training schedule
- Visualization and logging options

---
File path: `configs/underwater/deeplabv3_r50_Underwater_AAU.py`

```python
# ============================================================
# Custom DeepLabV3 (ResNet-50) for Underwater Dataset
# ============================================================

# Base configurations
_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',          # Model definition
    '../_base_/datasets/turbidity_underwater_dataset.py',  # Dataset + dataloaders
    '../_base_/default_runtime.py',                  # Logging, hooks, runtime
    '../_base_/schedules/schedule_160k.py'           # Training schedule (iterations)
]

# Optional: custom module imports (e.g., for transforms)
custom_imports = dict(imports=['mmseg.datasets.transforms'], allow_failed_imports=False)

# ------------------------------------------------------------
# Model Override
# ------------------------------------------------------------
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    # Preprocessing for input images
    data_preprocessor=dict(
        size=crop_size,          # Input image crop/resize size
        size_divisor=None        # Optional padding divisor
    ),
    # Adjust number of output classes to match dataset
    decode_head=dict(num_classes=10, ignore_index=255),
    auxiliary_head=dict(num_classes=10, ignore_index=255)
)

# ------------------------------------------------------------
# Visualization & Logging
# ------------------------------------------------------------
vis_backends = [dict(type='TensorboardVisBackend')]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Custom logging and checkpoint hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=16000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
```

## Training a model in MMSegmentation

```python
cd mmsegmentation
source .venv/bin/activate
# optional run with tmux so it will run in the background when remote shell closed
# Select which GPU to use (e.g., GPU 0)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/underwater/custom_deeplabv3_r50.py
```

# Visualize training with TensorBoard.

## 📡 SSH Port Forwarding and TensorBoard Access

When training models remotely (e.g., on a Jetson, HPC node, or lab server), you can securely view **TensorBoard** in your **local browser** using SSH port forwarding.

### 1️⃣ SSH Port Forwarding Basics

SSH port forwarding allows you to create a secure tunnel from your **local machine** to a **remote port** (e.g., TensorBoard’s default port `6006`).

```bash
ssh -L 6006:localhost:6006 <username>@<remote_server_ip>
```

Go to project folder, source and execute tensorboard.

```bash
cd ~/mmsegmentation
source .venv/bin/activate
tensorboard --logdir work_dirs --bind_all
```
Open `localhost:6006` in local machine to access TensorBoard and visualize the training.

# How to make inference from a trained data.


Make an inference from your testing data. This testing data need to be define in the dataloader of the model you are using. Snippet of this code.

```python
...
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',  # no annotations needed
        data_prefix=dict(
            img_path='test/' + turbidity_level + '/image',
            seg_map_path='test/' + turbidity_level + '/mask'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1280, 720), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ],
    ),
)

...
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    output_dir='work_dirs/vis_preds',
    format_only=False
)
```
Make an inference. First parameter `configs/turbidity_training/t1_mask2former_swin-s.py` makes a reference to the config file you used to trani the model. While `work_dirs/t1_mask2former_swin/s/iter_19000.pth` is the weightpoint generated while training.

Use the weightpoints that has converfeg and not overfit the model (check that the model loss funcion is decreasing while the validation metrics are not being degraded).

```bash
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/turbidity_training/t1_mask2former_swin-s.py work_dirs/t1_mask2former_swin-s/iter_19000.pth --show-dir work_dirs/vis_preds
```

The previous command generate a file in `~/mmsegmentation/work_dirs/vis_preds`. The prediction generates a grayscale masks (hard to visualize). They can easily be visualize in RGB with the GroungTruth and prediction by using Tensorboard (explianed in this same document).

To mIoU, preccision, recall and dice you can run the next script.

```python

import os
import cv2
import numpy as np
import csv
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================

# Ground truth (GT) and prediction directories
gt_dir = '/home/ubuntu/data/test/1/mask'       # path to GT grayscale masks
pred_dir = 'work_dirs/vis_preds'               # path to predicted grayscale masks

# Class definitions (update if needed)
classes = [
    'Pool_background', 'Black_Pipe', 'Aluminum_Pipe', 'Wooden_piece', 'Granite',
    'Acomar', 'Can', 'Benchy', 'Intruder', 'BlueROV2'
]
num_classes = len(classes)

# Optional: output CSV for reporting
csv_out = 'work_dirs/metrics_summary.csv'

# ============================================================
# LOAD DATA
# ============================================================

gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

all_gt, all_pred = [], []

for g, p in zip(gt_files, pred_files):
    gt = cv2.imread(os.path.join(gt_dir, g), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(os.path.join(pred_dir, p), cv2.IMREAD_GRAYSCALE)
    if gt is None or pr is None:
        print(f"[Warning] Skipped {g} or {p} (not readable).")
        continue
    all_gt.append(gt.flatten())
    all_pred.append(pr.flatten())

all_gt = np.concatenate(all_gt)
all_pred = np.concatenate(all_pred)

# ============================================================
# METRIC COMPUTATION
# ============================================================

cm = confusion_matrix(all_gt, all_pred, labels=range(num_classes))
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP

# Avoid division by zero
precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
recall    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
f1        = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(TP, dtype=float), where=(precision + recall) != 0)
iou       = np.divide(TP, TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN) != 0)

# Overall (macro) metrics
overall_precision = precision.mean()
overall_recall    = recall.mean()
overall_f1        = f1.mean()
overall_iou       = iou.mean()

# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 72)
print(f"{'Class':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8}")
print("-" * 72)
for i, name in enumerate(classes):
    print(f"{name:<20} {precision[i]:8.3f} {recall[i]:8.3f} {f1[i]:8.3f} {iou[i]:8.3f}")
print("-" * 72)
print(f"{'Mean (m)':<20} {overall_precision:8.3f} {overall_recall:8.3f} {overall_f1:8.3f} {overall_iou:8.3f}")
print("=" * 72)

# ============================================================
# OPTIONAL: SAVE TO CSV
# ============================================================

os.makedirs(os.path.dirname(csv_out), exist_ok=True)
with open(csv_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'IoU'])
    for i, name in enumerate(classes):
        writer.writerow([name, precision[i], recall[i], f1[i], iou[i]])
    writer.writerow(['Mean', overall_precision, overall_recall, overall_f1, overall_iou])

print(f"\n✅ Metrics saved to: {csv_out}")
```

This will print in the screen all the metrics of all the classes and the average of all the classes.

To obtain a confusion matrix of prediction run next script!

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================
gt_dir = '/home/ubuntu/data/test/1/mask'        # Ground truth grayscale masks
pred_dir = 'work_dirs/vis_preds'                # Predicted grayscale masks
save_path = 'work_dirs/confusion_matrix.png'    # Output image

classes = [
    'Pool_background', 'Black_Pipe', 'Aluminum_Pipe', 'Wooden_piece', 'Granite',
    'Acomar', 'Can', 'Benchy', 'Intruder', 'BlueROV2'
]
num_classes = len(classes)

# ============================================================
# LOAD MASKS
# ============================================================
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

all_gt, all_pred = [], []

for g, p in zip(gt_files, pred_files):
    gt = cv2.imread(os.path.join(gt_dir, g), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(os.path.join(pred_dir, p), cv2.IMREAD_GRAYSCALE)
    if gt is None or pr is None:
        print(f"[Warning] Skipped {g} or {p} (not readable).")
        continue
    all_gt.append(gt.flatten())
    all_pred.append(pr.flatten())

all_gt = np.concatenate(all_gt)
all_pred = np.concatenate(all_pred)

# ============================================================
# COMPUTE CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(all_gt, all_pred, labels=range(num_classes))
cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)  # normalize rows

# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={'label': 'Normalized Frequency'}
)

plt.title("Normalized Confusion Matrix (Prediction vs. Ground Truth)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n✅ Confusion matrix saved as:", save_path)
print("\nRaw counts (first few rows):")
print(cm[:min(5, num_classes), :min(5, num_classes)])

```

Tachannnnnn, you got it! now go and write the damm paper!




<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_demo.svg)](https://openxlab.org.cn/apps?search=mmseg)

Documentation: <https://mmsegmentation.readthedocs.io/en/latest/>

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch works with PyTorch 1.6+.

### 🎉 Introducing MMSegmentation v1.0.0 🎉

We are thrilled to announce the official release of MMSegmentation's latest version! For this new release, the [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch serves as the primary branch, while the development branch is [dev-1.x](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x). The stable branch for the previous release remains as the [0.x](https://github.com/open-mmlab/mmsegmentation/tree/0.x) branch. Please note that the [master](https://github.com/open-mmlab/mmsegmentation/tree/master) branch will only be maintained for a limited time before being removed. We encourage you to be mindful of branch selection and updates during use. Thank you for your unwavering support and enthusiasm, and let's work together to make MMSegmentation even more robust and powerful! 💪

MMSegmentation v1.x brings remarkable improvements over the 0.x release, offering a more flexible and feature-packed experience. To utilize the new features in v1.x, we kindly invite you to consult our detailed [📚 migration guide](https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html), which will help you seamlessly transition your projects. Your support is invaluable, and we eagerly await your feedback!

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## What's New

v1.2.0 was released on 10/12/2023, from 1.1.0 to 1.2.0, we have added or updated the following features:

### Highlights

- Support for the open-vocabulary semantic segmentation algorithm [SAN](configs/san/README.md)

- Support monocular depth estimation task, please refer to [VPD](configs/vpd/README.md) and [Adabins](projects/Adabins/README.md) for more details.

  ![depth estimation](https://github.com/open-mmlab/mmsegmentation/assets/15952744/07afd0e9-8ace-4a00-aa1e-5bf0ca92dcbc)

- Add new projects: open-vocabulary semantic segmentation algorithm [CAT-Seg](projects/CAT-Seg/README.md), real-time semantic segmentation algofithm [PP-MobileSeg](projects/pp_mobileseg/README.md)

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb) on Colab.

To migrate from MMSegmentation 0.x, please refer to [migration](docs/en/migration).

## Tutorial

<div align="center">
  <b>MMSegmentation Tutorials</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Get Started</b>
      </td>
      <td>
        <b>MMSeg Basic Tutorial</b>
      </td>
      <td>
        <b>MMSeg Detail Tutorial</b>
      </td>
      <td>
        <b>MMSeg Development Tutorial</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="docs/en/overview.md">MMSeg overview</a></li>
          <li><a href="docs/en/get_started.md">MMSeg Installation</a></li>
          <li><a href="docs/en/notes/faq.md">FAQ</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/user_guides/1_config.md">Tutorial 1: Learn about Configs</a></li>
          <li><a href="docs/en/user_guides/2_dataset_prepare.md">Tutorial 2: Prepare datasets</a></li>
          <li><a href="docs/en/user_guides/3_inference.md">Tutorial 3: Inference with existing models</a></li>
          <li><a href="docs/en/user_guides/4_train_test.md">Tutorial 4: Train and test with existing models</a></li>
          <li><a href="docs/en/user_guides/5_deployment.md">Tutorial 5: Model deployment</a></li>
          <li><a href="docs/zh_cn/user_guides/deploy_jetson.md">Deploy mmsegmentation on Jetson platform</a></li>
          <li><a href="docs/en/user_guides/useful_tools.md">Useful Tools</a></li>
          <li><a href="docs/en/user_guides/visualization_feature_map.md">Feature Map Visualization</a></li>
          <li><a href="docs/en/user_guides/visualization.md">Visualization</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/advanced_guides/datasets.md">MMSeg Dataset</a></li>
          <li><a href="docs/en/advanced_guides/models.md">MMSeg Models</a></li>
          <li><a href="docs/en/advanced_guides/structures.md">MMSeg Dataset Structures</a></li>
          <li><a href="docs/en/advanced_guides/transforms.md">MMSeg Data Transforms</a></li>
          <li><a href="docs/en/advanced_guides/data_flow.md">MMSeg Dataflow</a></li>
          <li><a href="docs/en/advanced_guides/engine.md">MMSeg Training Engine</a></li>
          <li><a href="docs/en/advanced_guides/evaluation.md">MMSeg Evaluation</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/advanced_guides/add_datasets.md">Add New Datasets</a></li>
          <li><a href="docs/en/advanced_guides/add_metrics.md">Add New Metrics</a></li>
          <li><a href="docs/en/advanced_guides/add_models.md">Add New Modules</a></li>
          <li><a href="docs/en/advanced_guides/add_transforms.md">Add New Data Transforms</a></li>
          <li><a href="docs/en/advanced_guides/customize_runtime.md">Customize Runtime Settings</a></li>
          <li><a href="docs/en/advanced_guides/training_tricks.md">Training Tricks</a></li>
          <li><a href=".github/CONTRIBUTING.md">Contribute code to MMSeg</a></li>
          <li><a href="docs/zh_cn/advanced_guides/contribute_dataset.md">Contribute a standard dataset in projects</a></li>
          <li><a href="docs/en/device/npu.md">NPU (HUAWEI Ascend)</a></li>
          <li><a href="docs/en/migration/interface.md">0.x → 1.x migration</a></li>
          <li><a href="docs/en/migration/package.md">0.x → 1.x package</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Supported backbones</b>
      </td>
      <td>
        <b>Supported methods</b>
      </td>
      <td>
        <b>Supported Head</b>
      </td>
      <td>
        <b>Supported datasets</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="mmseg/models/backbones/resnet.py">ResNet(CVPR'2016)</a></li>
        <li><a href="mmseg/models/backbones/resnext.py">ResNeXt (CVPR'2017)</a></li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/mobilenet_v2">MobileNetV2 (CVPR'2018)</a></li>
        <li><a href="configs/mobilenet_v3">MobileNetV3 (ICCV'2019)</a></li>
        <li><a href="configs/vit">Vision Transformer (ICLR'2021)</a></li>
        <li><a href="configs/swin">Swin Transformer (ICCV'2021)</a></li>
        <li><a href="configs/twins">Twins (NeurIPS'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/poolformer">PoolFormer (CVPR'2022)</a></li>
        <li><a href="configs/segnext">SegNeXt (NeurIPS'2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/san/">SAN (CVPR'2023)</a></li>
          <li><a href="configs/vpd">VPD (ICCV'2023)</a></li>
          <li><a href="configs/ddrnet">DDRNet (T-ITS'2022)</a></li>
          <li><a href="configs/pidnet">PIDNet (ArXiv'2022)</a></li>
          <li><a href="configs/mask2former">Mask2Former (CVPR'2022)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/knet">K-Net (NeurIPS'2021)</a></li>
          <li><a href="configs/segformer">SegFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/segmenter">Segmenter (ICCV'2021)</a></li>
          <li><a href="configs/dpt">DPT (ArXiv'2021)</a></li>
          <li><a href="configs/setr">SETR (CVPR'2021)</a></li>
          <li><a href="configs/stdc">STDC (CVPR'2021)</a></li>
          <li><a href="configs/bisenetv2">BiSeNetV2 (IJCV'2021)</a></li>
          <li><a href="configs/cgnet">CGNet (TIP'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/dnlnet">DNLNet (ECCV'2020)</a></li>
          <li><a href="configs/ocrnet">OCRNet (ECCV'2020)</a></li>
          <li><a href="configs/isanet">ISANet (ArXiv'2019/IJCV'2021)</a></li>
          <li><a href="configs/fastscnn">Fast-SCNN (ArXiv'2019)</a></li>
          <li><a href="configs/fastfcn">FastFCN (ArXiv'2019)</a></li>
          <li><a href="configs/gcnet">GCNet (ICCVW'2019/TPAMI'2020)</a></li>
          <li><a href="configs/ann">ANN (ICCV'2019)</a></li>
          <li><a href="configs/emanet">EMANet (ICCV'2019)</a></li>
          <li><a href="configs/ccnet">CCNet (ICCV'2019)</a></li>
          <li><a href="configs/dmnet">DMNet (ICCV'2019)</a></li>
          <li><a href="configs/sem_fpn">Semantic FPN (CVPR'2019)</a></li>
          <li><a href="configs/danet">DANet (CVPR'2019)</a></li>
          <li><a href="configs/apcnet">APCNet (CVPR'2019)</a></li>
          <li><a href="configs/nonlocal_net">NonLocal Net (CVPR'2018)</a></li>
          <li><a href="configs/encnet">EncNet (CVPR'2018)</a></li>
          <li><a href="configs/deeplabv3plus">DeepLabV3+ (CVPR'2018)</a></li>
          <li><a href="configs/upernet">UPerNet (ECCV'2018)</a></li>
          <li><a href="configs/icnet">ICNet (ECCV'2018)</a></li>
          <li><a href="configs/psanet">PSANet (ECCV'2018)</a></li>
          <li><a href="configs/bisenetv1">BiSeNetV1 (ECCV'2018)</a></li>
          <li><a href="configs/deeplabv3">DeepLabV3 (ArXiv'2017)</a></li>
          <li><a href="configs/pspnet">PSPNet (CVPR'2017)</a></li>
          <li><a href="configs/erfnet">ERFNet (T-ITS'2017)</a></li>
          <li><a href="configs/unet">UNet (MICCAI'2016/Nat. Methods'2019)</a></li>
          <li><a href="configs/fcn">FCN (CVPR'2015/TPAMI'2017)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="mmseg/models/decode_heads/ann_head.py">ANN_Head</li>
          <li><a href="mmseg/models/decode_heads/apc_head.py">APC_Head</li>
          <li><a href="mmseg/models/decode_heads/aspp_head.py">ASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/cc_head.py">CC_Head</li>
          <li><a href="mmseg/models/decode_heads/da_head.py">DA_Head</li>
          <li><a href="mmseg/models/decode_heads/ddr_head.py">DDR_Head</li>
          <li><a href="mmseg/models/decode_heads/dm_head.py">DM_Head</li>
          <li><a href="mmseg/models/decode_heads/dnl_head.py">DNL_Head</li>
          <li><a href="mmseg/models/decode_heads/dpt_head.py">DPT_HEAD</li>
          <li><a href="mmseg/models/decode_heads/ema_head.py">EMA_Head</li>
          <li><a href="mmseg/models/decode_heads/enc_head.py">ENC_Head</li>
          <li><a href="mmseg/models/decode_heads/fcn_head.py">FCN_Head</li>
          <li><a href="mmseg/models/decode_heads/fpn_head.py">FPN_Head</li>
          <li><a href="mmseg/models/decode_heads/gc_head.py">GC_Head</li>
          <li><a href="mmseg/models/decode_heads/ham_head.py">LightHam_Head</li>
          <li><a href="mmseg/models/decode_heads/isa_head.py">ISA_Head</li>
          <li><a href="mmseg/models/decode_heads/knet_head.py">Knet_Head</li>
          <li><a href="mmseg/models/decode_heads/lraspp_head.py">LRASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/mask2former_head.py">mask2former_Head</li>
          <li><a href="mmseg/models/decode_heads/maskformer_head.py">maskformer_Head</li>
          <li><a href="mmseg/models/decode_heads/nl_head.py">NL_Head</li>
          <li><a href="mmseg/models/decode_heads/ocr_head.py">OCR_Head</li>
          <li><a href="mmseg/models/decode_heads/pid_head.py">PID_Head</li>
          <li><a href="mmseg/models/decode_heads/point_head.py">point_Head</li>
          <li><a href="mmseg/models/decode_heads/psa_head.py">PSA_Head</li>
          <li><a href="mmseg/models/decode_heads/psp_head.py">PSP_Head</li>
          <li><a href="mmseg/models/decode_heads/san_head.py">SAN_Head</li>
          <li><a href="mmseg/models/decode_heads/segformer_head.py">segformer_Head</li>
          <li><a href="mmseg/models/decode_heads/segmenter_mask_head.py">segmenter_mask_Head</li>
          <li><a href="mmseg/models/decode_heads/sep_aspp_head.py">SepASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/sep_fcn_head.py">SepFCN_Head</li>
          <li><a href="mmseg/models/decode_heads/setr_mla_head.py">SETRMLAHead_Head</li>
          <li><a href="mmseg/models/decode_heads/setr_up_head.py">SETRUP_Head</li>
          <li><a href="mmseg/models/decode_heads/stdc_head.py">STDC_Head</li>
          <li><a href="mmseg/models/decode_heads/uper_head.py">Uper_Head</li>
          <li><a href="mmseg/models/decode_heads/vpd_depth_head.py">VPDDepth_Head</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#cityscapes">Cityscapes</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-voc">PASCAL VOC</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#ade20k">ADE20K</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context">Pascal Context</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-10k">COCO-Stuff 10k</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k">COCO-Stuff 164k</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#chase-db1">CHASE_DB1</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#drive">DRIVE</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hrf">HRF</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#stare">STARE</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#dark-zurich">Dark Zurich</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nighttime-driving">Nighttime Driving</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#loveda">LoveDA</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam">Potsdam</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-vaihingen">Vaihingen</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid">iSAID</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#mapillary-vistas-datasets">Mapillary Vistas</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#levir-cd">LEVIR-CD</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#bdd100K">BDD100K</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nyu">NYU</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hsi-drive-2.0">HSIDrive20</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Supported loss</b></li>
        <ul>
          <li><a href="mmseg/models/losses/boundary_loss.py">boundary_loss</a></li>
          <li><a href="mmseg/models/losses/cross_entropy_loss.py">cross_entropy_loss</a></li>
          <li><a href="mmseg/models/losses/dice_loss.py">dice_loss</a></li>
          <li><a href="mmseg/models/losses/focal_loss.py">focal_loss</a></li>
          <li><a href="mmseg/models/losses/huasdorff_distance_loss.py">huasdorff_distance_loss</a></li>
          <li><a href="mmseg/models/losses/kldiv_loss.py">kldiv_loss</a></li>
          <li><a href="mmseg/models/losses/lovasz_loss.py">lovasz_loss</a></li>
          <li><a href="mmseg/models/losses/ohem_cross_entropy_loss.py">ohem_cross_entropy_loss</a></li>
          <li><a href="mmseg/models/losses/silog_loss.py">silog_loss</a></li>
          <li><a href="mmseg/models/losses/tversky_loss.py">tversky_loss</a></li>
        </ul>
        </ul>
      </td>
  </tbody>
</table>

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Projects

[Here](projects/README.md) are some implementations of SOTA models and solutions built on MMSegmentation, which are supported and maintained by community users. These projects demonstrate the best practices based on MMSegmentation for research and product development. We welcome and appreciate all the contributions to OpenMMLab ecosystem.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## OpenMMLab Family

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
