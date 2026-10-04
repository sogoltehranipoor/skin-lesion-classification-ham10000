# skin-lesion-classification-ham10000
Description: Skin lesion classification on HAM10000 (7 classes) using MobileNetV2 transfer learning, built in Google Colab
# Skin Lesion Classification (HAM10000)

Classifying dermatoscopic images into 7 skin lesion types using transfer learning.

> ⚠️ Educational project only. Not a medical diagnostic tool.

## Dataset
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): 10,015 images, 7 classes
(akiec, bcc, bkl, df, mel, nv, vasc). Highly imbalanced (~67% nv).

## Approach
1. Train/val split by `lesion_id` to avoid data leakage
2. Data augmentation (flips, brightness, contrast)
3. MobileNetV2 (ImageNet weights): feature extraction, then fine-tuning
4. Softened class weights (sqrt of balanced) for the imbalance
5. EarlyStopping + ReduceLROnPlateau

## Results
| Stage | Accuracy | Macro F1 | Melanoma recall |
|---|---|---|---|
| Feature extraction | 0.59 | 0.44 | 0.43 |
| After fine-tuning | ... | ... | ... |

![confusion matrix](images/confusion_matrix.png)

## Challenges & what I learned
- Class imbalance: accuracy is misleading, so I tracked macro F1 and melanoma recall
- Data leakage from multiple images per lesion
- Fine-tuning with a low learning rate and frozen BatchNorm layers

## How to run
Open the notebook in Colab, enable GPU, and run all cells.
The dataset downloads automatically via `kagglehub`.
