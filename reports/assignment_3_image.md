# Assignment 3: Image Data Report

## 1. Problem Statement
The objective of this notebook is to develop an automated computer vision classification system for recyclable garbage and municipal waste sorting. Given an input image of waste material, the model categorizes it into one of six material recycling classes.

- **Task Type:** Multi-class image classification (6 classes).
- **Target Classes:**
  - `0`: `cardboard`
  - `1`: `glass`
  - `2`: `metal`
  - `3`: `paper`
  - `4`: `plastic`
  - `5`: `trash`
- **Core Challenges:**
  - **Visual intraclass variability:** Objects appear in diverse shapes, deformation states (e.g., crushed aluminum cans vs intact cans), orientations, and background lighting conditions.
  - **Interclass visual similarity:** Visual overlap exists between visually similar categories, such as thin cardboard versus thick paper, or clear plastic bottles versus transparent glassware.
  - **Small-scale dataset:** With only ~2,500 total images, training a convolutional network from scratch risks severe overfitting, making transfer learning and data augmentation indispensable.

---

## 2. Dataset Information
- **Dataset Name:** Garbage Classification
- **Source:** Kaggle ([Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification))
- **Total Images:** 2,527 images.
- **Image Specifications:** 3-channel RGB images, original resolution $512 \times 384$ pixels.

### Dataset Splits & Class Distribution

| Class Label | Training Set (~70%) | Validation Set (~15%) | Test Set (~15%) | Total per Class |
| :--- | :---: | :---: | :---: | :---: |
| `paper` | 416 | 89 | 89 | 594 |
| `glass` | 350 | 75 | 76 | 501 |
| `plastic` | 337 | 72 | 73 | 482 |
| `metal` | 287 | 61 | 62 | 410 |
| `cardboard` | 282 | 61 | 60 | 403 |
| `trash` | 96 | 21 | 20 | 137 |
| **Total** | **1,768** | **379** | **380** | **2,527** |

### Exploratory Data Analysis & Augmentations
- **Class Imbalance:** `trash` is the most underrepresented category (96 training images), while `paper` is the most frequent (416 images).
- **Data Augmentation (Train):**
  - `transforms.Resize((224, 224))`
  - `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))`
  - `transforms.RandomHorizontalFlip(p=0.5)`
  - `transforms.RandomRotation(degrees=15)`
  - `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet standard)
- **Validation & Test Transforms:** Resize to $224 \times 224$, CenterCrop, and ImageNet Normalization.

---

## 3. Algorithms & Feature Extraction

The notebook compares two distinct modeling strategies across **5 modern pre-trained convolutional neural network (CNN) backbones**:
- `resnet18`
- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `efficientnet_b3`

### Strategy A: Feature Extraction + Traditional Classifiers (30 Configurations)
Feature vectors are extracted from the global pooling layer of each frozen backbone network, followed by training traditional classifiers:
1. **Logistic Regression (`LogisticRegression`):**
   - Regularization penalty $\in \{\text{"l2"}, \text{None}\}$, `max_iter=500` (10 configurations).
2. **Random Forest (`RandomForestClassifier`):**
   - Number of trees $n\_estimators \in \{100, 150\}$, `random_state=42` (10 configurations).
3. **Support Vector Machine (`SVC`):**
   - Penalty parameter $C \in \{1, 10\}$, RBF kernel (10 configurations).

### Strategy B: Deep Transfer Learning & Fine-Tuning (10 Configurations)
Using PyTorch with custom classification heads:
1. **Head Classifier Fine-Tuning (`head_classifier`):**
   - Pre-trained backbone parameters are completely frozen; only the custom classification head (`Linear` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(6)`) is trained (5 configurations).
2. **End-to-End Fine-Tuning (`end_to_end`):**
   - Both the pre-trained backbone layers and the classification head are trained jointly end-to-end (5 configurations).
- *Training Setup:* CrossEntropyLoss, Adam optimizer ($\text{lr}=10^{-4}$ for end-to-end, $10^{-3}$ for head-only), batch size 32, 5 epochs.

---

## 4. Evaluation Measures & Results

### Evaluation Metrics
Evaluated on the held-out test set (380 images):
- **Accuracy:** Overall proportion of correct waste class predictions.
- **Macro Precision:** Arithmetic mean of precision scores across all 6 waste classes.
- **Macro Recall:** Arithmetic mean of recall scores across all 6 waste classes.
- **Macro F1-Score:** Primary ranking metric; gives equal weight to all classes including minority classes (`trash`).

### Best Configuration per Model Family

| Model Family | Best Model Configuration | Pretrained Backbone | Accuracy | Precision | Recall | Macro F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Learning (MLP)** | **`end_to_end` fine-tuning** | **`efficientnet_b3`** | **0.9132** | **0.9113** | **0.8962** | **0.9026** |
| **SVM** | $C = 1$, RBF kernel | `resnet50` | 0.8947 | 0.9015 | 0.8710 | 0.8825 |
| **Logistic Regression** | `penalty = 'l2'` | `resnet50` | 0.8974 | 0.8847 | 0.8764 | 0.8781 |
| **Random Forest** | $n\_estimators = 150$ | `resnet50` | 0.8026 | 0.8656 | 0.7583 | 0.7968 |

### Complete Results Table (All 40 Models)

| # | Backbone | Classifier / Head | Model Config | Accuracy | Precision | Recall | Macro F1 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `resnet18` | Logistic Regression | l2 | 0.82 | 0.83 | 0.79 | 0.80 |
| 1 | `resnet34` | Logistic Regression | l2 | 0.85 | 0.86 | 0.83 | 0.84 |
| 2 | `resnet50` | Logistic Regression | l2 | 0.90 | 0.88 | 0.88 | **0.88** |
| 3 | `efficientnet_b0` | Logistic Regression | l2 | 0.89 | 0.90 | 0.86 | 0.88 |
| 4 | `efficientnet_b3` | Logistic Regression | l2 | 0.87 | 0.87 | 0.84 | 0.85 |
| 5 | `resnet18` | Logistic Regression | None | 0.80 | 0.80 | 0.77 | 0.78 |
| 6 | `resnet34` | Logistic Regression | None | 0.82 | 0.84 | 0.80 | 0.81 |
| 7 | `resnet50` | Logistic Regression | None | 0.90 | 0.89 | 0.87 | 0.87 |
| 8 | `efficientnet_b0` | Logistic Regression | None | 0.87 | 0.86 | 0.85 | 0.86 |
| 9 | `efficientnet_b3` | Logistic Regression | None | 0.87 | 0.86 | 0.84 | 0.85 |
| 10 | `resnet18` | Random Forest | 100 | 0.76 | 0.82 | 0.73 | 0.76 |
| 11 | `resnet34` | Random Forest | 100 | 0.75 | 0.81 | 0.73 | 0.75 |
| 12 | `resnet50` | Random Forest | 100 | 0.78 | 0.86 | 0.75 | 0.78 |
| 13 | `efficientnet_b0` | Random Forest | 100 | 0.79 | 0.87 | 0.77 | 0.79 |
| 14 | `efficientnet_b3` | Random Forest | 100 | 0.74 | 0.85 | 0.72 | 0.74 |
| 15 | `resnet18` | Random Forest | 150 | 0.76 | 0.82 | 0.73 | 0.76 |
| 16 | `resnet34` | Random Forest | 150 | 0.75 | 0.82 | 0.73 | 0.75 |
| 17 | `resnet50` | Random Forest | 150 | 0.80 | 0.86 | 0.77 | **0.80** |
| 18 | `efficientnet_b0` | Random Forest | 150 | 0.77 | 0.87 | 0.76 | 0.77 |
| 19 | `efficientnet_b3` | Random Forest | 150 | 0.74 | 0.85 | 0.72 | 0.74 |
| 20 | `resnet18` | SVM | 1 | 0.80 | 0.82 | 0.79 | 0.80 |
| 21 | `resnet34` | SVM | 1 | 0.78 | 0.79 | 0.77 | 0.78 |
| 22 | `resnet50` | SVM | 1 | 0.88 | 0.90 | 0.87 | **0.88** |
| 23 | `efficientnet_b0` | SVM | 1 | 0.88 | 0.90 | 0.87 | 0.88 |
| 24 | `efficientnet_b3` | SVM | 1 | 0.86 | 0.87 | 0.85 | 0.86 |
| 25 | `resnet18` | SVM | 10 | 0.80 | 0.82 | 0.79 | 0.80 |
| 26 | `resnet34` | SVM | 10 | 0.78 | 0.79 | 0.77 | 0.78 |
| 27 | `resnet50` | SVM | 10 | 0.88 | 0.90 | 0.87 | **0.88** |
| 28 | `efficientnet_b0` | SVM | 10 | 0.88 | 0.90 | 0.87 | 0.88 |
| 29 | `efficientnet_b3` | SVM | 10 | 0.86 | 0.87 | 0.85 | 0.86 |
| 30 | `resnet18` | Deep Learning (MLP) | `end_to_end` | 0.7658 | 0.8064 | 0.6937 | 0.7038 |
| 31 | `resnet34` | Deep Learning (MLP) | `end_to_end` | 0.7711 | 0.8286 | 0.6793 | 0.6854 |
| 32 | `resnet50` | Deep Learning (MLP) | `end_to_end` | 0.8237 | 0.8605 | 0.7462 | 0.7553 |
| 33 | `efficientnet_b0` | Deep Learning (MLP) | `end_to_end` | 0.9000 | 0.9156 | 0.8872 | 0.8982 |
| **34** | **`efficientnet_b3`** | **Deep Learning (MLP)** | **`end_to_end`** | **0.9132** | **0.9113** | **0.8962** | **0.9026** |
| 35 | `resnet18` | Deep Learning (MLP) | `head_classifier` | 0.7579 | 0.7986 | 0.7217 | 0.7394 |
| 36 | `resnet34` | Deep Learning (MLP) | `head_classifier` | 0.7684 | 0.7896 | 0.7351 | 0.7502 |
| 37 | `resnet50` | Deep Learning (MLP) | `head_classifier` | 0.8605 | 0.8830 | 0.8256 | 0.8444 |
| 38 | `efficientnet_b0` | Deep Learning (MLP) | `head_classifier` | 0.8789 | 0.8998 | 0.8427 | 0.8621 |
| 39 | `efficientnet_b3` | Deep Learning (MLP) | `head_classifier` | 0.8474 | 0.8317 | 0.8062 | 0.8156 |

### Top 5 Models Across All Experiments

1. **`efficientnet_b3` / Deep MLP (`end_to_end`):** Macro F1 = **0.9026** (Accuracy: 91.32%)
2. **`efficientnet_b0` / Deep MLP (`end_to_end`):** Macro F1 = **0.8982** (Accuracy: 90.00%)
3. **`resnet50` / SVM ($C=1$ or $10$):** Macro F1 = **0.8825** (Accuracy: 88.42%)
4. **`efficientnet_b0` / SVM ($C=1$ or $10$):** Macro F1 = **0.8805** (Accuracy: 88.16%)
5. **`resnet50` / Logistic Regression (`penalty=l2`):** Macro F1 = **0.8781** (Accuracy: 89.74%)

### Key Findings & Conclusion
- **Overall Champion:** **EfficientNet-B3 trained end-to-end** achieved the highest performance across all evaluation metrics, reaching **91.32% test accuracy** and **0.9026 Macro F1-score**.
- **Architecture Comparison:** EfficientNet architectures (`b0` and `b3`) consistently outperformed the ResNet family when fine-tuned end-to-end, benefiting from compound scaling and mobile inverted bottleneck convolutions.
- **Traditional vs. Deep Models:** Traditional models (SVM and Logistic Regression) trained on pre-extracted ResNet50 and EfficientNet-B0 embeddings proved to be extremely competitive (Macro F1 ~ 0.88), training in seconds while approaching the accuracy of full deep fine-tuning.
