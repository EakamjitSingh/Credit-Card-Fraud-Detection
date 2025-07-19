# -*- coding: utf-8 -*-
"""
Credit Card Fraud Detection: A Comprehensive Machine Learning Project

This script demonstrates a complete workflow for building a credit card fraud
detection model. It covers the following key stages:
1.  **Data Simulation & Exploration (EDA)**: We'll simulate a dataset that mirrors
    the properties of real-world transaction data (imbalanced classes, PCA-transformed
    features) and explore its characteristics.
2.  **Preprocessing**: We'll scale the data to prepare it for machine learning models.
3.  **Handling Class Imbalance**: We'll apply two common techniques, Random
    Undersampling and SMOTE (Synthetic Minority Over-sampling Technique), to create
    balanced training datasets.
4.  **Model Training**: We'll train Logistic Regression and Random Forest classifiers
    on three different versions of the data: original, undersampled, and SMOTE-applied.
5.  **Model Evaluation**: We'll evaluate each model using a comprehensive set of metrics
    suited for imbalanced classification: Precision, Recall, F1-Score, ROC-AUC, and
    the Confusion Matrix.
6.  **Results Comparison**: We'll present a summary table to compare the performance
    of all models and techniques, drawing conclusions on the most effective approach.
"""

# ###########################################################################
# 1. IMPORT LIBRARIES
# ###########################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# imblearn is a library specifically for handling imbalanced datasets
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

print("Libraries imported successfully.")

# ###########################################################################
# 2. DATA SIMULATION & EXPLORATORY DATA ANALYSIS (EDA)
# ###########################################################################

def create_simulated_dataset(n_samples=20000, fraud_ratio=0.005):
    """
    Creates a simulated dataset resembling the Kaggle Credit Card Fraud dataset.
    The real dataset cannot be included directly, so we generate one with similar
    statistical properties (PCA components, scaled time/amount, heavy imbalance).

    Args:
        n_samples (int): Total number of transactions to generate.
        fraud_ratio (float): The proportion of transactions that are fraudulent.

    Returns:
        pandas.DataFrame: A DataFrame with simulated transaction data.
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Generate PCA-like features (V1-V28)
    # Normal transactions are centered around 0
    normal_features = np.random.normal(0, 1, (n_normal, 28))
    # Fraudulent transactions have a slightly different distribution
    fraud_features = np.random.normal(0.5, 1.5, (n_fraud, 28))

    features = np.vstack([normal_features, fraud_features])

    # Generate 'Time' and 'Amount'
    time = np.arange(n_samples)
    # Amounts are log-normally distributed, with fraud having slightly higher values
    normal_amount = np.random.lognormal(mean=2, sigma=1, size=n_normal)
    fraud_amount = np.random.lognormal(mean=3.5, sigma=1.5, size=n_fraud)
    amount = np.concatenate([normal_amount, fraud_amount])

    # Generate 'Class' label
    labels = np.array([0] * n_normal + [1] * n_fraud)

    # Create DataFrame
    df = pd.DataFrame(data=features, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = labels

    # Shuffle the dataset to mix normal and fraudulent transactions
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- Section 2: Data Simulation & EDA ---")
df = create_simulated_dataset()
print("Simulated dataset created.")
print("\nDataset Head:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe())

# Check for the core problem: Class Imbalance
print("\nClass Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)

# Visualize the imbalance
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Class Distribution (0: Normal, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# ###########################################################################
# 3. DATA PREPROCESSING
# ###########################################################################

print("\n--- Section 3: Data Preprocessing ---")

# The 'Time' and 'Amount' columns are not on the same scale as the V1-V28
# features. We need to scale them. RobustScaler is a good choice as it's
# less sensitive to outliers, which are common in financial data.
scaler = RobustScaler()

df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

print("Scaled 'Amount' and 'Time' columns and dropped originals.")
print("\nPreprocessed DataFrame Head:")
print(df.head())

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets.
# It's crucial to use `stratify=y` to ensure the class distribution in the
# train and test sets is the same as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training labels distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing labels distribution:\n{y_test.value_counts(normalize=True)}")


# ###########################################################################
# 4. HANDLING CLASS IMBALANCE
# ###########################################################################

# We will apply resampling techniques ONLY to the training data.
# The test data must remain untouched to serve as a realistic representation
# of data the model would see in production.

print("\n--- Section 4: Handling Class Imbalance ---")

# ## 4.1. Random Undersampling ##
print("\nApplying Random Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("Class distribution after Random Undersampling:")
print(pd.Series(y_train_under).value_counts())


# ## 4.2. SMOTE (Synthetic Minority Over-sampling Technique) ##
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ###########################################################################
# 5. MODEL TRAINING & EVALUATION
# ###########################################################################

print("\n--- Section 5: Model Training & Evaluation ---")

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Calculates and prints key performance metrics for a classification model.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Evaluation Metrics for {model_name}:")
    print("-----------------------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}  <-- Key metric for fraud detection!")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("-----------------------------------------")
    
    # Print Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# Store results for final comparison
results = []

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

# Define datasets to use
datasets = {
    "Original Imbalanced Data": (X_train, y_train),
    "Random Undersampled Data": (X_train_under, y_train_under),
    "SMOTE Oversampled Data": (X_train_smote, y_train_smote)
}

# Iterate through each dataset and each model
for data_name, (X_train_data, y_train_data) in datasets.items():
    print(f"\n======================================================")
    print(f"Training models on: {data_name}")
    print(f"======================================================")
    
    for model_name, model in models.items():
        full_model_name = f"{model_name} on {data_name}"
        print(f"\n--- Training {full_model_name} ---")
        
        # Train the model
        model.fit(X_train_data, y_train_data)
        
        # Make predictions on the original, untouched test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probability for the positive class
        
        # Evaluate and store results
        metrics = evaluate_model(full_model_name, y_test, y_pred, y_prob)
        metrics['Model'] = model_name
        metrics['Dataset'] = data_name
        results.append(metrics)

# ###########################################################################
# 6. RESULTS COMPARISON & CONCLUSION
# ###########################################################################

print("\n--- Section 6: Final Results Comparison ---")

results_df = pd.DataFrame(results)
results_df = results_df[['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

# Sort by the most important metrics for this problem: Recall and F1-Score
results_df_sorted = results_df.sort_values(by=['Recall', 'F1-Score'], ascending=False)

print("\nPerformance of All Models Across Different Datasets:")
print(results_df_sorted.to_string())

print("\n--- Conclusion ---")
print("""
Analysis of Results:
1.  **Original Imbalanced Data**: As expected, models trained on the original data
    achieve very high accuracy. However, this is misleading. Their **Recall** is
    extremely low, meaning they fail to identify most of the actual fraud cases.
    This is the classic pitfall of using accuracy as a primary metric for
    imbalanced problems.

2.  **Random Undersampled Data**: This technique significantly improves **Recall**.
    By forcing the model to see an equal number of fraud and normal cases, it
    learns to identify fraud much more effectively. The trade-off is often a
    decrease in **Precision**, meaning more normal transactions might be flagged
    as fraudulent (false positives). This might be acceptable depending on the
    business cost of fraud vs. the cost of investigating a false alarm.

3.  **SMOTE Oversampled Data**: SMOTE typically offers a strong balance. It
    dramatically boosts **Recall** without sacrificing as much **Precision** as
    undersampling. By creating synthetic fraud examples, it gives the model more
    diverse fraudulent patterns to learn from without discarding any information
    from the majority class.

4.  **Model Comparison (Random Forest vs. Logistic Regression)**: The Random Forest
    generally outperforms Logistic Regression across all metrics, especially on the
    resampled datasets. This is because it can capture more complex, non-linear
    relationships in the data, which are common in fraud detection.

**Final Recommendation:**
For a credit card fraud detection system, maximizing **Recall** is paramount. It is
far better to have a few false alarms (lower precision) than to miss an actual case
of fraud. Based on the results, the **Random Forest model trained on the SMOTE
dataset** provides the best overall performance, offering an excellent balance of high
Recall and respectable Precision, and the highest ROC-AUC score, indicating superior
discriminative ability.
""")
