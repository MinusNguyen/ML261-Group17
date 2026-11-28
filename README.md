# Machine Learning (ML261) — Group 17

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Traditional%20ML-F7931E?logo=scikit-learn&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Features%20Tracked-F05032?logo=git&logoColor=white)

Course project repository for **Machine Learning (ML261 — Semester 261)** at Ho Chi Minh City University of Technology (HCMUT — VNU-HCM). This repository contains exploratory data analysis (EDA), preprocessing pipelines, pre-extracted feature representations, traditional machine learning and deep learning experiments, and evaluation reports across three data modalities: **Tabular**, **Text**, and **Image** classification.

---

## Team Members

| No. | Full Name | Student ID | GitHub | Email |
| :-: | :--- | :---: | :--- | :--- |
| 1 | `Nguyễn Phúc Nguyên` | `2352831` | [@MinusNguyen](https://github.com/MinusNguyen) | `nguyen.nguyenfeliks0612@hcmut.edu.vn` |
| 2 | `Nguyễn Thành Nguyên` | `2352832` | [@FrogKingDead](https://github.com/FrogKingDead) | `nguyen.nguyenthanhk23@hcmut.edu.vn` |
| 3 | `Đinh Ngọc Việt` | `2353317` | [@GemVii](https://github.com/GemVii) | `viet.dinhngoc2005@hcmut.edu.vn` |
| 4 | `Lê Minh Trí` | `2353228` | [@TriLM2](https://github.com/TriLM2) | `tri.leminh21050@hcmut.edu.vn` |

---

## Repository Structure

```text
ML261-Group17/
├── .gitattributes                          # Git LFS tracking rules for features/**
├── README.md                               # Project overview, dependencies, and usage guide
├── dataset/
│   └── dataset.md                          # Schemas, splits, and characteristics of all 3 datasets
├── features/                               # Pre-extracted features (~783 MB, tracked via Git LFS)
│   ├── assignment_1/
│   │   ├── mild/                           # Mildly balanced tabular features (.npy) & metadata (.csv)
│   │   └── none/                           # Unbalanced tabular features (.npy) & metadata (.csv)
│   ├── assignment_2/                       # Sparse TF-IDF (.npz), TruncatedSVD (.npy), BERT (.npy)
│   └── assignment_3/                       # CNN embeddings & labels (train / valid / test)
│       ├── efficientnet_b0/
│       ├── efficientnet_b3/
│       ├── resnet18/
│       ├── resnet34/
│       └── resnet50/
├── notebooks/
│   ├── assignment_1_tabular.ipynb          # Tabular classification (BRFSS 2023 Diabetes)
│   ├── assignment_2_text.ipynb             # Text classification (Food Delivery App Reviews)
│   └── assignment_3_image.ipynb            # Image classification (Garbage Classification)
└── reports/
    ├── assignment_1_tabular.md             # Assignment 1 methodology & results report
    ├── assignment_2_text.md                # Assignment 2 methodology & results report
    └── assignment_3_image.md               # Assignment 3 methodology & results report
```

---

## Assignments & Results Overview

Detailed dataset schemas and class distributions are documented in [`dataset/dataset.md`](dataset/dataset.md).

| Assignment | Modality | Dataset | Classes | Experiment Scope | Best Model & Highlight Metrics | Files |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **Assignment 1** | **Tabular** | Diabetes Health Indicators (BRFSS 2023) — 261,589 rows, 23 cols | 3 (`No Diabetes`, `Pre-diabetes`, `Diabetes`) | 8 preprocessing grids × 6 model families (KNN, Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch MLP) | **Best Accuracy:** XGBoost (`lr=0.05`, `balance=none`, `ordinal`, no PCA) — **82.55% Acc**, 0.3931 Macro F1, 0.7806 Weighted F1<br>**Best Macro F1:** PyTorch MLP (`hidden=2`, `balance=mild`, `onehot`, no PCA) — **0.4493 Macro F1**, 77.72% Acc, 0.7678 Weighted F1 | [Notebook](notebooks/assignment_1_tabular.ipynb) \| [Report](reports/assignment_1_tabular.md) |
| **Assignment 2** | **Text** | Food Delivery Apps Reviews — 1,247,061 raw reviews (100,000 stratified subsample: 75k train / 25k test) | 5 (`1` to `5` stars) | 30 runs across 5 feature sets (`tfidf_2500`, `tfidf_5000`, `svd_2500`, `svd_5000`, `bert-base-uncased`) and 5 models (Logistic Regression, Linear SVC, Random Forest, XGBoost, PyTorch MLP) | **Best Macro F1:** Logistic Regression (`C=2.0`, `bert`) — **0.4305 Macro F1**, 77.93% Acc, 0.7370 Weighted F1<br>**Best Accuracy & Weighted F1:** PyTorch MLP (`bert`) — **78.76% Acc**, **0.7437 Weighted F1**, 0.4303 Macro F1 | [Notebook](notebooks/assignment_2_text.ipynb) \| [Report](reports/assignment_2_text.md) |
| **Assignment 3** | **Image** | Garbage Classification — 2,527 RGB images (512×384; 1,768 train / 379 valid / 380 test) | 6 (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) | 40 configs across 5 `timm` CNN backbones (`resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`, `efficientnet_b3`): 30 traditional ML + 10 deep fine-tuning | **Best Overall:** EfficientNet-B3 (`end_to_end` fine-tuning) — **91.32% Acc**, **0.9026 Macro F1** (0.9113 Precision, 0.8962 Recall)<br>**Best Traditional ML:** ResNet-50 + SVM (RBF) — **0.8825 Macro F1**; ResNet-50 + Logistic Regression (`l2`) — **89.74% Acc** (0.8781 Macro F1) | [Notebook](notebooks/assignment_3_image.ipynb) \| [Report](reports/assignment_3_image.md) |

---

## Pre-Extracted Features (Git LFS)

Binary feature files under `features/` (~783 MB total) are tracked with **Git LFS** (`.gitattributes`).

| Directory | Splits & Sample Counts | Feature Representations & Shapes | Notes |
| :--- | :--- | :--- | :--- |
| `features/assignment_1/mild/` | Train: `106,228` \| Test: `48,688` | `engineered` (`46-d`), `top20` (`20-d`), `pca95` (`22-d`), `ae32` (`32-d`), `y_train.npy`, `y_test.npy`, `feature_names.csv`, `feature_importances.csv` | Additional engineered/reduced/autoencoder feature sets; `assignment_1_tabular.ipynb` runs its own 8-grid `DiabetesDataPipeline` directly from the Kaggle CSV |
| `features/assignment_1/none/` | Train: `194,752` \| Test: `48,688` | Same representations as `mild/` with natural class distribution | Same as above |
| `features/assignment_2/` | Train: `75,000` \| Test: `25,000` | `tfidf_2500` (`.npz` sparse, 2500-d), `tfidf_5000` (`.npz` sparse, 5000-d), `svd_2500` (`300-d`), `svd_5000` (`300-d`), `bert` (`768-d` `bert-base-uncased`) | Reused by `assignment_2_text.ipynb` when `EXTRACT_FEATURE = False` |
| `features/assignment_3/<backbone>/` | Train: `1,768` \| Valid: `379` \| Test: `380` | `resnet18` (`512-d`), `resnet34` (`512-d`), `resnet50` (`2048-d`), `efficientnet_b0` (`1280-d`), `efficientnet_b3` (`1536-d`) + `*_labels.npy` | Extracted from frozen `timm` backbones for traditional ML & `head_classifier` training |

---

## Dependencies & Environment

### System Requirements
- **Python 3.10+** (developed and tested on Google Colab)
- **NVIDIA GPU (e.g., T4)** recommended for Assignment 2 (BERT feature extraction & PyTorch MLP) and Assignment 3 (CNN feature extraction & end-to-end fine-tuning)
- **Git LFS** required to pull binary `.npy` and `.npz` files under `features/`

### Python Packages

| Category | Package | Used In | Purpose |
| :--- | :--- | :--- | :--- |
| **Core & Visualization** | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` | A1, A2, A3 | Numerical arrays, tabular manipulation, sparse matrices, EDA plots |
| **Data & Notebook** | `kagglehub`, `tqdm`, `jupyter` / `ipython` | A1, A2, A3 | Automated Kaggle dataset downloads, progress bars, notebook execution |
| **Traditional ML** | `scikit-learn` | A1, A2, A3 | Preprocessing, PCA, TF-IDF, SVD, KNN, LR, LinearSVC/SVC, RF, metrics |
| **Gradient Boosting** | `xgboost` | A1, A2 | `XGBClassifier` models |
| **Gradient Boosting** | `lightgbm` | A1 | `LGBMClassifier` models |
| **Deep Learning & Vision** | `torch` | A1, A2, A3 | PyTorch MLPs and CNN fine-tuning |
| **Vision Models & I/O** | `torchvision`, `timm`, `Pillow` | A3 | Image transforms, `ImageFolder`, pretrained ResNet & EfficientNet backbones |
| **NLP & Text** | `transformers`, `nltk`, `wordcloud` | A2 | `bert-base-uncased` embeddings, lemmatization/stopwords, word clouds |

### Local Installation

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost lightgbm \
            torch torchvision timm Pillow transformers nltk wordcloud kagglehub tqdm jupyter
```

> **Notes:**
> - On **Google Colab**, each notebook includes inline `!pip install` cells for packages not pre-installed in the default runtime (e.g., `kagglehub`, `lightgbm`, `timm`, `transformers`, `nltk`, `xgboost`, `wordcloud`).
> - `assignment_2_text.ipynb` automatically downloads the required NLTK corpora (`stopwords`, `wordnet`, `omw-1.4`).

---

## Getting Started

### 1. Clone the Repository (with Git LFS)

```bash
git lfs install
git clone https://github.com/MinusNguyen/ML261-Group17.git
cd ML261-Group17
git lfs pull
```

### 2. Run the Notebooks

#### Option A — Google Colab (Recommended)
1. Upload or open any notebook from `notebooks/` in Google Colab.
2. Enable a GPU runtime (`Runtime` -> `Change runtime type` -> `T4 GPU`) for **Assignment 2** and **Assignment 3**.
3. Run cells sequentially. Datasets are automatically fetched via `kagglehub`.

#### Option B — Local Jupyter Environment
1. Create and activate a Python 3.10+ virtual environment, then install the dependencies above.
2. Launch Jupyter (`jupyter notebook` or `jupyter lab`) and open the target notebook under `notebooks/`.
3. **Path note for Assignment 3:** `notebooks/assignment_3_image.ipynb` defaults to Google Colab paths (`/content/datasets` and `/content/features`). Update these directory variables to local paths when running outside Colab.

### 3. Control Flags

Each notebook exposes top-level boolean flags to skip heavy feature extraction or retraining when pre-extracted features or saved result CSVs are already available:

| Notebook | Flags | Behavior & Output File |
| :--- | :--- | :--- |
| `assignment_1_tabular.ipynb` | `TRAIN_MODEL` | Set `True` to train all configurations and save `model_result_diabetes_012_v3.csv`; set `False` to load saved results. |
| `assignment_2_text.ipynb` | `EXTRACT_FEATURE`, `TRAIN_MODEL` | Uses `./features/` (or `ML_Assig2_FoodReviews_Features.zip`); saves/loads `model_result_food_reviews.csv`. |
| `assignment_3_image.ipynb` | `EXTRACT_FEATURES`, `TRAIN_MODEL` | Uses `/content/features/` (or `features_garbage.zip`); saves/loads `model_result_garbage.csv`. |

---

## Documentation & Reports

- **Dataset Documentation:** [dataset/dataset.md](dataset/dataset.md)
- **Assignment 1 Report (Tabular):** [reports/assignment_1_tabular.md](reports/assignment_1_tabular.md)
- **Assignment 2 Report (Text):** [reports/assignment_2_text.md](reports/assignment_2_text.md)
- **Assignment 3 Report (Image):** [reports/assignment_3_image.md](reports/assignment_3_image.md)
