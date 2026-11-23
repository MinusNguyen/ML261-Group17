# Datasets Documentation

This document describes the three datasets used across **Tabular**, **Text**, and **Image** classification tasks.

## Summary

| Modality | Dataset | Kaggle Source | Target / Labels | Classes |
| :--- | :--- | :--- | :--- | :---: |
| **Tabular** | Diabetes Health Indicators Dataset (BRFSS 2023) | [`luisangelparra/diabetes-health-indicators-dataset`](https://www.kaggle.com/datasets/luisangelparra/diabetes-health-indicators-dataset) | `Diabetes_012` (No Diabetes, Pre-diabetes, Diabetes) | 3 |
| **Text** | Food Delivery Apps Reviews | [`skamlo/food-delivery-apps-reviews`](https://www.kaggle.com/datasets/skamlo/food-delivery-apps-reviews) | `score` (1 to 5 stars) | 5 |
| **Image** | Garbage Classification | [`asdasdasasdas/garbage-classification`](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) | Waste material category | 6 |

---

## 1. Diabetes Health Indicators Dataset (BRFSS 2023)

* **Modality:** Tabular (CSV)
* **Source:** [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/luisangelparra/diabetes-health-indicators-dataset)
* **File Used:** `diabetes_012_health_indicators_BRFSS2023.csv`
* **Origin:** Cleaned health survey responses (~261,589 records) derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2023.

### Target Variable

* `Diabetes_012`:
  * `0`: No Diabetes (~83% of records)
  * `1`: Pre-diabetes (~2.5% of records)
  * `2`: Diabetes (~14.6% of records)

### Feature Categories

All columns are numerically encoded and grouped into three main types:

* **Continuous Numerical:**
  * `BMI`: Body Mass Index (contains extreme right-tail values).
  * `MentHlth`: Number of days with poor mental health in the past 30 days (0–30).
  * `PhysHlth`: Number of days with poor physical health in the past 30 days (0–30).
* **Ordinal:**
  * `AgeGroup`: 13-level age bracket (`1` = 18–24 up to `13` = 80+).
  * `GenHlth`: Self-reported general health (`1` = Excellent to `5` = Poor).
  * `Education`: Highest completed education level.
  * `Income`: Household income bracket.
* **Binary / Health & Lifestyle Indicators:**
  * Clinical conditions and mobility: `HighBP`, `HighChol`, `HeartDiseaseorAttack`, `Stroke`, `DiffWalk`, `KidneyDisease`, `Asthma`, `COPD`.
  * Lifestyle and healthcare access: `Smoker`, `PhysActivity`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`.

### Key Data Characteristics

* **Class Imbalance:** Heavily skewed toward Class `0` (No Diabetes), with Class `1` (Pre-diabetes) representing a very small minority.
* **Completeness & Duplicates:** The cleaned BRFSS 2023 file contains no missing values, though duplicate survey response rows exist.
* **Encoding Notes:** Certain BRFSS medical columns (such as `KidneyDisease`, `Asthma`, and `COPD`) may use `{1 = Yes, 2 = No}` survey coding rather than `{0, 1}`.

---

## 2. Food Delivery Apps Reviews

* **Modality:** Text (CSV)
* **Source:** [Kaggle - Food Delivery Apps Reviews](https://www.kaggle.com/datasets/skamlo/food-delivery-apps-reviews)
* **Origin:** User reviews and star ratings scraped from the **Google Play Store** and **Apple App Store** for major food delivery applications (collected February 12–14, 2024).

### Schema

| Column | Description | Notes |
| :--- | :--- | :--- |
| `content` | Published text of the user review | Primary text input |
| `score` | Star rating awarded by the user (`1` to `5`) | Target variable |
| `date` | Local date and time the review was posted | Metadata |
| `userName` | Display name of the reviewer | Metadata |
| `app` | Name of the food delivery application | Metadata |
| `platform` | Mobile store (`Google Play` or `App Store`) | Metadata |
| `appVersion` | Version of the mobile application reviewed | Metadata |
| `replyContent` | Developer response text to the review | Rarely populated (mostly missing) |
| `repliedDate` | Date the developer replied to the review | Rarely populated (mostly missing) |

### Key Data Characteristics

* **Skewed Ratings:** Like most app store review datasets, ratings are heavily concentrated at the extremes (`1 star` and `5 stars`), with fewer middle ratings (`2`, `3`, and `4 stars`).
* **Sparse Columns:** Columns related to developer replies (`replyContent`, `repliedDate`) are missing for the majority of rows and are typically ignored.
* **Informal Text:** Reviews vary widely in length—from single-word comments to multi-sentence complaints—and include punctuation, emojis, slang, and occasional URLs or mentions.

---

## 3. Garbage Classification

* **Modality:** Image (RGB JPEG)
* **Source:** [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
* **Size:** 2,527 labeled RGB images organized into 6 class-named directories.

### Classes

| Class | Description |
| :--- | :--- |
| `cardboard` | Cardboard boxes, packaging, and corrugated paperboard |
| `glass` | Glass bottles, jars, and containers |
| `metal` | Aluminum cans, foil, and scrap metal items |
| `paper` | Sheets of paper, newspapers, magazines, and paper bags |
| `plastic` | Plastic bottles, containers, bags, and cups |
| `trash` | Non-recyclable general household waste |

### Key Data Characteristics

* **Class Imbalance:** `paper` is the largest category, while `trash` is significantly smaller than the five recyclable classes.
* **Folder Layout:** Images are stored directly inside one folder per class without predefined `train`, `validation`, or `test` splits.
* **Visual Composition:** Photographs feature single waste items captured under varied lighting, angles, and mostly plain/neutral backgrounds.