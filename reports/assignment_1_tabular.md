# Assignment 1: Tabular Data Report

## 1. Problem Statement
The objective of this notebook is to solve a multi-class medical diagnosis prediction task: predicting whether an individual has **no diabetes**, **pre-diabetes**, or **diabetes** based on demographic, lifestyle, and clinical health indicators.

- **Task Type:** Multi-class classification (3 target classes).
- **Target Classes:**
  - `0`: No diabetes
  - `1`: Pre-diabetes
  - `2`: Diabetes
- **Core Challenges:**
  - **Severe class imbalance:** Pre-diabetes cases constitute a very small percentage compared to non-diabetic and diagnosed diabetic populations.
  - **Feature heterogeneity:** Mix of continuous biometric measurements, binary health indicators, and multi-level ordinal survey scales.
  - **Evaluation nuance:** A naïve model predicting the majority class could achieve high raw accuracy while failing to detect at-risk pre-diabetic patients. Therefore, balanced class evaluation (Macro F1) is paramount alongside Accuracy.

---

## 2. Dataset Information
- **Dataset Name:** Diabetes Health Indicators Dataset (BRFSS 2023 - `diabetes_012`)
- **Source:** Kaggle ([Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/luisangelparra/diabetes-health-indicators-dataset)), derived from the CDC's Behavioral Risk Factor Surveillance System (BRFSS 2023).
- **Data File:** `diabetes_012_health_indicators_BRFSS2023.csv`
- **Total Records & Features:** 261,589 rows, 23 columns.
- **Missing Values:** None (0 missing values across all columns in BRFSS 2023).

### Feature Breakdown
1. **Target Column (1):**
   - `Diabetes_012`: `0` (No diabetes), `1` (Pre-diabetes), `2` (Diabetes).
2. **Continuous Numerical Features (3):**
   - `BMI`: Body Mass Index (Mean: 28.82, Std: 6.52, Range: 12.0 – 100.0).
   - `MentHlth`: Days of poor mental health in past 30 days (Mean: 4.18, Range: 0 – 30).
   - `PhysHlth`: Days of physical illness or injury in past 30 days (Mean: 4.40, Range: 0 – 30).
3. **Categorical / Binary / Ordinal Features (19):**
   - Clinical & Health: `KidneyDisease`, `HighBP`, `HighChol`, `CholCheck`, `Asthma`, `COPD`, `Stroke`, `HeartDiseaseorAttack`, `GenHlth` (General health scale 1–5), `DiffWalk` (Difficulty walking).
   - Lifestyle & Behavioral: `Smoker`, `PhysActivity`, `HvyAlcoholConsump`.
   - Healthcare Access: `AnyHealthcare`, `NoDocbcCost`.
   - Demographics: `Sex` (0: Female, 1: Male), `AgeGroup` (14-level age brackets), `Education` (6-level education attainment), `Income` (11-level income tiers).

### Data Splitting & Class Balancing
- **Train/Test Split:** 80% training / 20% testing (stratified by `Diabetes_012`).
  - **Test Set:** 48,688 samples.
  - **Training Set:** 194,752 samples (`balance_mode = "none"`) or 106,228 samples (`balance_mode = "mild"`, which mitigates majority-class dominance).

---

## 3. Algorithms & Preprocessing Pipeline

### Preprocessing Experimentation Grid
A combinatorial grid of $2 \times 2 \times 2 = 8$ preprocessing pipelines was tested:
1. **Balancing Strategy (`balance_mode`):**
   - `"none"`: Natural class distribution preserved (train shape: 194,752).
   - `"mild"`: Downsamples the majority class to balance learning (train shape: 106,228).
2. **Categorical Encoding (`cat_encode`):**
   - `"ordinal"`: Preserves integer order (feature count: 20–26 dimensions).
   - `"onehot"`: One-Hot Encoding of categorical variables (feature count expands up to 48 dimensions).
3. **Dimensionality Reduction (`pca_variance`):**
   - `0.95`: Principal Component Analysis retaining 95% cumulative explained variance.
   - `None`: Retains all raw encoded features.
- *Numerical Scaling:* Applied `StandardScaler` to continuous features across all configurations.

### Algorithms Implemented
The notebook evaluated both traditional machine learning models (using Scikit-Learn, XGBoost, and LightGBM) and deep learning models (using PyTorch):

1. **K-Nearest Neighbors (`KNeighborsClassifier`):**
   - Hyperparameters: `n_neighbors` $\in \{25, 51\}$, `weights="distance"`.
2. **Logistic Regression (`LogisticRegression`):**
   - Hyperparameters: `penalty` $\in \{\text{None}, \text{"l2"}\}$, `max_iter=500`.
3. **Random Forest (`RandomForestClassifier`):**
   - Hyperparameters: `n_estimators` $\in \{150, 250\}$, `max_depth=14`, `min_samples_leaf=5`, `n_jobs=-1`.
4. **XGBoost (`XGBClassifier`):**
   - Hyperparameters: `learning_rate` $\in \{0.05, 0.1\}$, `n_estimators=350`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`.
5. **LightGBM (`LGBMClassifier`):**
   - Hyperparameters: `learning_rate` $\in \{0.05, 0.1\}$, `n_estimators=350`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`.
6. **Multi-Layer Perceptron (`MLP` via PyTorch):**
   - Architecture: Input Layer $\rightarrow$ Hidden Layers with `Linear` + `BatchNorm1d` + `ReLU` + `Dropout(0.2)` $\rightarrow$ Output Linear Layer (3 logits).
   - Hidden Layer Options: `num_hidden` $\in \{2, 3\}$.
   - Optimizer & Training: `AdamW` ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), `CosineAnnealingLR` scheduler, `CrossEntropyLoss`, batch size 256, 10 epochs.

---

## 4. Evaluation Measures & Results

### Evaluation Metrics
Because the dataset is heavily skewed toward Class 0 (no diabetes), evaluating solely on accuracy would hide severe under-performance on minority classes (pre-diabetes):
- **Accuracy:** Overall proportion of correctly classified instances.
- **Macro F1-Score:** Unweighted arithmetic mean of F1-scores across all 3 classes (critical for rewarding balanced detection of pre-diabetes).
- **Weighted F1-Score:** F1-score weighted by support of each class.

### Model Performance Summary

| Model | Best By Accuracy Config | Accuracy | Macro F1 | Weighted F1 | Best By Macro F1 Config | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors** | $k=51$, balance=none, encode=onehot, pca=None | 0.8153 | 0.3669 | 0.7647 | $k=51$, balance=mild, encode=ordinal, pca=None | 0.7628 | 0.4412 | 0.7584 |
| **Logistic Regression** | penalty=None, balance=none, encode=onehot, pca=None | 0.8216 | 0.3891 | 0.7772 | penalty=l2, balance=mild, encode=onehot, pca=None | 0.7838 | 0.4444 | 0.7712 |
| **Random Forest** | $n=150$, balance=none, encode=ordinal, pca=None | 0.8249 | 0.3813 | 0.7751 | $n=150$, balance=mild, encode=onehot, pca=None | 0.7840 | 0.4436 | 0.7709 |
| **XGBoost** | lr=0.05, balance=none, encode=ordinal, pca=None | **0.8255** | 0.3931 | 0.7806 | lr=0.10, balance=mild, encode=onehot, pca=None | 0.7775 | 0.4486 | 0.7681 |
| **LightGBM** | lr=0.05, balance=none, encode=ordinal, pca=None | 0.8254 | 0.3949 | 0.7814 | lr=0.05, balance=mild, encode=onehot, pca=None | 0.7794 | 0.4470 | 0.7691 |
| **PyTorch MLP** | hidden=3, balance=none, encode=ordinal, pca=None | 0.8252 | 0.3846 | 0.7766 | hidden=2, balance=mild, encode=onehot, pca=None | 0.7772 | **0.4493** | 0.7678 |

### Key Findings & Conclusion
- **Highest Raw Accuracy:** **XGBoost (82.55%)** with ordinal encoding, no class balancing, and raw feature space.
- **Highest Macro F1-Score:** **PyTorch MLP (0.4493)** with 2 hidden layers, one-hot encoding, mild class balancing, and raw feature space.
- **Balancing Trade-off:** `balance_mode="none"` favors peak overall accuracy (~82.5%) at the expense of poor pre-diabetes recall (Macro F1 < 0.40). In contrast, `balance_mode="mild"` elevates the Macro F1 significantly (from ~0.38 to ~0.45) with only a modest drop in accuracy (~77.7% to 78.4%), providing substantially better clinical utility.
- **Dimensionality Reduction:** Applying PCA (`pca_variance=0.95`) consistently yielded slightly lower performance than retaining the full feature set across all classifiers.
