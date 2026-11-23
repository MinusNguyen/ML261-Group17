# Assignment 2: Text Data Report

## 1. Problem Statement
The objective of this notebook is to perform sentiment and user satisfaction classification from raw textual customer reviews of popular food delivery applications. Specifically, the system predicts the user's star rating (1 to 5 stars) based on their written review text.

- **Task Type:** Multi-class text classification (5 classes).
- **Target Classes:**
  - `0`: 1 star
  - `1`: 2 stars
  - `2`: 3 stars
  - `3`: 4 stars
  - `4`: 5 stars
- **Core Challenges:**
  - **Extreme bimodal class imbalance:** Over 82% of all reviews are polarized into either 1-star (negative) or 5-star (positive) ratings, with 2, 3, and 4-star reviews representing a small minority.
  - **Noisy, informal text:** Reviews contain slang, misspellings, emojis, punctuation irregularities, and varying sentence lengths (from single-word feedback to paragraph-length complaints).
  - **Representation trade-offs:** Comparing high-dimensional sparse representations (TF-IDF), low-dimensional dense singular value decomposition (SVD), and deep contextual semantic embeddings (BERT).

---

## 2. Dataset Information
- **Dataset Name:** Food Delivery Apps Reviews
- **Source:** Kaggle ([Food Delivery Apps Reviews](https://www.kaggle.com/datasets/skamlo/food-delivery-apps-reviews))
- **Total Records in Raw Corpus:** 1,247,061 reviews.
- **Applications Covered:** Uber Eats (962,166), Grubhub (209,784), Glovo (47,758), Wolt (18,458), Bolt Food (8,895).
- **Platforms:** Google Play Store (1,146,101 reviews), Apple App Store (100,960 reviews).

### Original Full Distribution by Star Rating
- **1 star:** 386,150 reviews (30.96%)
- **2 stars:** 49,446 reviews (3.96%)
- **3 stars:** 58,431 reviews (4.69%)
- **4 stars:** 117,580 reviews (9.43%)
- **5 stars:** 635,454 reviews (50.96%)

### Stratified Subsample for Training & Evaluation
To ensure scalable feature extraction (especially for BERT and SVD on Colab GPUs/CPUs), the dataset is capped at **100,000 stratified samples**:
- `1 star`: 30,965
- `2 stars`: 3,965
- `3 stars`: 4,685
- `4 stars`: 9,429
- `5 stars`: 50,956
- **Splits:** 75,000 samples for training (further split 80/20 into train/validation for deep learning) and 25,000 samples for test evaluation.

---

## 3. Algorithms & Preprocessing Pipeline

### Text Cleaning & Preprocessing Tracks
Raw text underwent regex-based cleaning (lowercase conversion, removal of URLs, `@mentions`, emojis, special characters, punctuation, digits, and redundant whitespace). Two distinct processing tracks were prepared:
- **`processed_0` (Stopwords Removed):** Standard English stopwords eliminated for sparse statistical n-gram representations.
- **`processed_1` (Stopwords Retained):** Syntactic structure and stop words preserved for contextual transformer-based embeddings.

### Feature Extraction Configurations (5 Configurations)
1. **`tfidf_2500`:** TF-IDF Vectorizer with `max_features = 2500`, sublinear term frequency scaling, and unigram + bigram range `(1, 2)`.
2. **`tfidf_5000`:** TF-IDF Vectorizer with `max_features = 5000`, sublinear term frequency scaling, and unigram + bigram range `(1, 2)`.
3. **`svd_2500`:** TruncatedSVD (`n_components = 300`) fitted on the `tfidf_2500` sparse matrix.
4. **`svd_5000`:** TruncatedSVD (`n_components = 300`) fitted on the `tfidf_5000` sparse matrix.
5. **`bert`:** Pre-trained transformer embeddings from `bert-base-uncased` (768-dimensional dense vectors derived via mean pooling over token embeddings with `max_length = 128`).

### Algorithms Implemented (30 Configurations Total)
1. **Logistic Regression (`LogisticRegression`):**
   - Regularization strength $C \in \{0.5, 1.0, 2.0\}$ across `tfidf_2500`, `tfidf_5000`, and `bert` (9 configurations).
2. **Linear Support Vector Classifier (`LinearSVC`):**
   - Regularization parameter $C \in \{0.5, 1.0, 2.0\}$ across `tfidf_2500`, `tfidf_5000`, and `bert` (9 configurations).
3. **Random Forest (`RandomForestClassifier`):**
   - Number of trees $n\_estimators \in \{50, 100\}$ on dense SVD matrices `svd_2500` and `svd_5000` (4 configurations).
4. **XGBoost (`XGBClassifier`):**
   - Number of boosting rounds $n\_estimators \in \{50, 100\}$ on `svd_2500`, `svd_5000`, and `bert` (6 configurations).
5. **Deep Learning Multi-Layer Perceptron (`MLP` via PyTorch):**
   - Architecture on BERT (768 features): `Linear(768, 256)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, 32)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(32, 5)`.
   - Architecture on SVD (300 features): `Linear(300, 128)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128, 32)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(32, 5)`.
   - Optimization: `AdamW` ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), `StepLR` scheduler (`step_size=10`, $\gamma=0.5$), `CrossEntropyLoss`, early stopping with patience 10 on validation loss.

---

## 4. Evaluation Measures & Results

### Evaluation Metrics
Given the 5-class distribution with heavy rating polarization:
- **Accuracy:** Overall multi-class classification accuracy.
- **Macro Precision & Macro Recall:** Unweighted average precision and recall across all 5 classes.
- **Macro F1-Score:** Primary benchmark metric; penalizes models that ignore intermediate classes (2, 3, and 4 stars).
- **Weighted F1-Score:** F1-score weighted by the true class proportions.

### Best Configuration per Model Type

| Model | Best Configuration | Feature Representation | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $C = 2.0$ | **BERT** | 0.7793 | 0.5012 | 0.4294 | **0.4305** | 0.7370 |
| **Linear SVC** | $C = 2.0$ | **TF-IDF 5000** | 0.7726 | 0.4914 | 0.4216 | **0.4241** | 0.7307 |
| **PyTorch MLP** | Dropout + StepLR + Early Stopping | **BERT** | **0.7876** | 0.4775 | 0.4324 | **0.4303** | **0.7437** |
| **XGBoost** | $n\_estimators = 100$ | **BERT** | 0.7720 | 0.4799 | 0.4160 | **0.4132** | 0.7270 |
| **Random Forest** | $n\_estimators = 100$ | **SVD 5000** | 0.7534 | 0.4531 | 0.3761 | **0.3503** | 0.6884 |

### Comprehensive Results Comparison (All 30 Runs)

| # | Classifier | Model Parameter | Feature / Preprocessing | Accuracy | Macro F1 | Weighted F1 |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Logistic Regression | $C = 0.5$ | `tfidf_2500` | 0.7785 | 0.4084 | 0.7276 |
| 2 | Logistic Regression | $C = 0.5$ | `tfidf_5000` | 0.7799 | 0.4032 | 0.7275 |
| 3 | Logistic Regression | $C = 0.5$ | `bert` | 0.7795 | 0.4285 | 0.7365 |
| 4 | Logistic Regression | $C = 1.0$ | `tfidf_2500` | 0.7790 | 0.4187 | 0.7315 |
| 5 | Logistic Regression | $C = 1.0$ | `tfidf_5000` | 0.7807 | 0.4177 | 0.7333 |
| 6 | Logistic Regression | $C = 1.0$ | `bert` | 0.7794 | 0.4298 | 0.7368 |
| 7 | Logistic Regression | $C = 2.0$ | `tfidf_2500` | 0.7782 | 0.4257 | 0.7341 |
| 8 | Logistic Regression | $C = 2.0$ | `tfidf_5000` | 0.7804 | 0.4289 | 0.7372 |
| **9** | **Logistic Regression** | **$C = 2.0$** | **`bert`** | **0.7793** | **0.4305** | **0.7370** |
| 10 | Linear SVC | $C = 0.5$ | `tfidf_2500` | 0.7763 | 0.4044 | 0.7253 |
| 11 | Linear SVC | $C = 0.5$ | `tfidf_5000` | 0.7774 | 0.4131 | 0.7295 |
| 12 | Linear SVC | $C = 0.5$ | `bert` | 0.7790 | 0.3934 | 0.7242 |
| 13 | Linear SVC | $C = 1.0$ | `tfidf_2500` | 0.7757 | 0.4076 | 0.7263 |
| 14 | Linear SVC | $C = 1.0$ | `tfidf_5000` | 0.7752 | 0.4192 | 0.7304 |
| 15 | Linear SVC | $C = 1.0$ | `bert` | 0.7789 | 0.3933 | 0.7242 |
| 16 | Linear SVC | $C = 2.0$ | `tfidf_2500` | 0.7752 | 0.4101 | 0.7268 |
| 17 | Linear SVC | $C = 2.0$ | `tfidf_5000` | 0.7726 | 0.4241 | 0.7307 |
| 18 | Linear SVC | $C = 2.0$ | `bert` | 0.7789 | 0.3933 | 0.7242 |
| 19 | Random Forest | $n = 50$ | `svd_2500` | 0.7484 | 0.3495 | 0.6848 |
| 20 | Random Forest | $n = 50$ | `svd_5000` | 0.7512 | 0.3499 | 0.6869 |
| 21 | Random Forest | $n = 100$ | `svd_2500` | 0.7511 | 0.3496 | 0.6863 |
| 22 | Random Forest | $n = 100$ | `svd_5000` | 0.7534 | 0.3503 | 0.6884 |
| 23 | XGBoost | $n = 50$ | `svd_2500` | 0.7596 | 0.3884 | 0.7082 |
| 24 | XGBoost | $n = 50$ | `svd_5000` | 0.7613 | 0.3897 | 0.7101 |
| 25 | XGBoost | $n = 50$ | `bert` | 0.7702 | 0.4093 | 0.7241 |
| 26 | XGBoost | $n = 100$ | `svd_2500` | 0.7645 | 0.3983 | 0.7153 |
| 27 | XGBoost | $n = 100$ | `svd_5000` | 0.7672 | 0.4029 | 0.7183 |
| 28 | XGBoost | $n = 100$ | `bert` | 0.7720 | 0.4132 | 0.7270 |
| **29** | **PyTorch MLP** | **Dropout + StepLR** | **`bert`** | **0.7876** | **0.4303** | **0.7437** |
| 30 | PyTorch MLP | Dropout + StepLR | `svd_5000` | 0.7710 | 0.3765 | 0.7126 |

### Key Findings & Conclusion
- **Highest Macro F1-Score:** **Logistic Regression ($C=2.0$ on BERT)** achieved the highest Macro F1 score of **0.4305**.
- **Highest Accuracy & Weighted F1:** **PyTorch MLP trained on BERT** achieved the highest overall Accuracy of **78.76%** and Weighted F1 of **0.7437**, with a Macro F1 of **0.4303** (virtually tied with Logistic Regression).
- **Embedding Superiority:** Models using contextual BERT embeddings consistently outperformed models relying on SVD-reduced features. Linear models (Logistic Regression and Linear SVC) on TF-IDF matrices also showed competitive results, while Random Forest on SVD performed the weakest due to information loss in truncated projection.
