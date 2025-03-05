import numpy as np
import cv2
import os
from glob import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

gt_folder = r"VOCdevkit/VOC2012/SegmentationClass"
pred_folder = "VOC2012_val_results"

# PASCAL VOC classes
NUM_CLASSES = 21

gt_paths = glob(os.path.join(gt_folder, "*.png"))
gt_filenames = [os.path.basename(p) for p in gt_paths]

gt_filenames = [f for f in gt_filenames if f"segmented_{f.replace('.png', '.jpg')}" in os.listdir(pred_folder)]

if not gt_filenames:
    print("Corresponding ground truth files not found")
    exit()

conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

def fast_hist(a, b, num_classes):
    k = (a >= 0) & (a < num_classes)
    return np.bincount(
        num_classes * a[k].astype(int) + b[k].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)

def cross_entropy_loss(gt_mask, pred_mask, num_classes):
    valid_mask = gt_mask != 255  
    gt_mask = np.clip(gt_mask, 0, num_classes - 1)
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)

    gt_one_hot = np.eye(num_classes)[gt_mask]
    pred_prob = np.eye(num_classes)[pred_mask]

    gt_one_hot = gt_one_hot[valid_mask]
    pred_prob = pred_prob[valid_mask]
    
    epsilon = 1e-10
    loss = -np.sum(gt_one_hot * np.log(pred_prob + epsilon)) / np.sum(valid_mask)
    return loss

loss_values = []

# Evaluation loop
for filename in gt_filenames:
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, f"segmented_{filename.replace('.png', '.jpg')}")

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    gt_mask[gt_mask == 255] = 0  

    pred_mask = np.clip(pred_mask, 0, NUM_CLASSES - 1)

    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    loss = cross_entropy_loss(gt_mask, pred_mask, NUM_CLASSES)
    loss_values.append(loss)

    conf_matrix += fast_hist(gt_mask.flatten(), pred_mask.flatten(), NUM_CLASSES)

# Plot Histogram of Cross-Entropy Loss
plt.figure(figsize=(8, 5))
plt.hist(loss_values, bins=20, color='b', alpha=0.7, edgecolor='black')
plt.xlabel("Cross-Entropy Loss")
plt.ylabel("Number of Images")
plt.title("Histogram of Cross-Entropy Loss for VOC")
plt.grid()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix[:20, :20], annot=False, cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix (First 20 Classes)")
plt.show()