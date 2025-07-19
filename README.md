# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project demonstrates a complete machine learning workflow for detecting fraudulent credit card transactions. Due to the highly imbalanced nature of transaction data (where fraudulent transactions are very rare), the project focuses on strategies to build effective classification models.

We explore the impact of different data sampling techniques—specifically **Random Undersampling** and **SMOTE (Synthetic Minority Over-sampling Technique)**—on the performance of two popular classification algorithms: **Logistic Regression** and **Random Forest**.

The ultimate goal is to identify the best-performing model that maximizes the detection of fraudulent transactions (Recall) while maintaining a reasonable level of precision.

---

## Features

- **Data Simulation**: Generates a realistic, imbalanced dataset with PCA-transformed features, mirroring common credit card fraud datasets.
- **Exploratory Data Analysis (EDA)**: Visualizes the severe class imbalance to establish a baseline.
- **Robust Preprocessing**: Uses `RobustScaler` to scale 'Time' and 'Amount' features, making the model less sensitive to outliers.
- **Advanced Imbalance Handling**: Implements and compares two key techniques:
    - **Random Undersampling**: Balances the dataset by reducing the number of non-fraudulent transactions.
    - **SMOTE**: Balances the dataset by generating synthetic fraudulent transactions.
- **Model Training**: Trains both Logistic Regression and Random Forest classifiers on three different datasets: the original imbalanced data, the undersampled data, and the SMOTE-resampled data.
- **Comprehensive Evaluation**: Evaluates models using metrics crucial for imbalanced classification:
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **ROC-AUC Score**
    - **Confusion Matrix**
- **Detailed Comparison**: Provides a summary table and a final conclusion comparing all six model-dataset combinations to identify the most effective strategy.

---

## Methodology

The project follows a structured data science workflow:

1.  **Data Preparation**: A simulated dataset of 20,000 transactions is created with a 0.5% fraud rate.
2.  **Preprocessing**: The `Time` and `Amount` features are scaled using `RobustScaler`. The data is then split into training and testing sets, ensuring the class distribution is maintained in both using `stratify`.
3.  **Resampling the Training Data**:
    - An undersampled dataset is created by randomly removing samples from the majority class (non-fraud).
    - A SMOTE-resampled dataset is created by generating new synthetic samples for the minority class (fraud).
    - **Important**: These resampling techniques are applied *only* to the training data to prevent data leakage and ensure the test set remains a realistic, unseen environment.
4.  **Modeling**: Logistic Regression and Random Forest models are trained on each of the three training datasets (original, undersampled, SMOTE).
5.  **Evaluation**: All trained models are tested on the original, untouched test set. Performance is measured, with a strong emphasis on **Recall**, as failing to detect a fraudulent transaction is the most costly error.
6.  **Conclusion**: The results are compiled and analyzed to determine the winning approach.

---

## How to Run This Project

### 1. Prerequisites

Ensure you have Python 3.6 or later installed on your system.

### 2. Clone the Repository (Optional)

If this project is on a platform like GitHub, you can clone it:
```bash
git clone <your-repository-url>
cd <repository-directory>
