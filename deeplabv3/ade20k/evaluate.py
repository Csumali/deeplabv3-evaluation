import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

gt_folder = "ADEChallengeData2016/annotations/validation"
pred_folder = "ADE20K_results"

gt_paths = sorted(glob.glob(os.path.join(gt_folder, "*.png")))
pred_paths = sorted(glob.glob(os.path.join(pred_folder, "*.png")))

assert len(gt_paths) == len(pred_paths), "Number of ground truth and predicted masks are not equal"

# ADE20K classes
num_classes = 150

loss_values = []

for gt_path, pred_path in zip(gt_paths, pred_paths):
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    pred_shape = pred_mask.shape
    gt_mask = cv2.resize(gt_mask, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)

    gt_mask = np.clip(gt_mask, 0, num_classes - 1)
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
    
    gt_one_hot = np.eye(num_classes)[gt_mask]
    pred_prob = np.eye(num_classes)[pred_mask]

    epsilon = 1e-10
    loss = -np.sum(gt_one_hot * np.log(pred_prob + epsilon)) / gt_mask.size
    loss_values.append(loss)

# Plot Histogram of Cross-Entropy Loss
plt.figure(figsize=(8, 5))
plt.hist(loss_values, bins=20, color='b', alpha=0.7, edgecolor='black')
plt.xlabel("Cross-Entropy Loss")
plt.ylabel("Number of Images")
plt.title("Histogram of Cross-Entropy Loss for ADE20K")
plt.grid()
plt.show()