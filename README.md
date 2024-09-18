# Breast Cancer Detection Using Machine Learning

This project compares four machine learning classification models—**K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Decision Tree**, and **Naive Bayes**—to predict whether a tumor is benign or malignant. The project utilizes the **Breast Cancer Wisconsin (Diagnostic)** dataset, which contains features related to the characteristics of cell nuclei present in breast cancer biopsies.

## Table of Contents
- Project Overview
- Dataset
- Models Compared
- Results
  - Naive Bayes
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- Conclusion
  - Key Insights

## Project Overview
The goal of this project is to compare the performance of four machine learning models on classifying breast cancer tumors as either benign or malignant. This is achieved using the **Breast Cancer Wisconsin (Diagnostic)** dataset, which includes features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset
The **Breast Cancer Wisconsin (Diagnostic)** dataset is a commonly used dataset for binary classification problems. It contains 569 samples and 30 numerical features that describe characteristics such as:
- **Radius**
- **Texture**
- **Perimeter**
- **Area**
- **Smoothness**

The target variable is a binary outcome:
- **0**: Malignant (cancerous)
- **1**: Benign (non-cancerous)


You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## Models Compared
The following machine learning classification models are compared:
1. **K-Nearest Neighbors (KNN)**: A simple and effective algorithm that classifies based on the majority vote of the nearest neighbors.
2. **Logistic Regression**: A statistical model that models the probability of a binary outcome based on linear relationships.
3. **Decision Tree**: A non-linear model that partitions the data into subsets based on feature values, using tree-like structures.
4. **Naive Bayes**: A probabilistic model based on Bayes' Theorem, assuming that features are conditionally independent.

## Results
After running the project, the performance of each model was evaluated based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**. Here are the results:

### Naive Bayes:
- **Accuracy**: 0.93
- **Precision**: 0.940
- **Recall**: 0.917
- **F1-Score**: 0.925

### Logistic Regression:
- **Accuracy**: 0.95
- **Precision**: 0.959
- **Recall**: 0.935
- **F1-Score**: 0.944

### K-Nearest Neighbors (KNN):
- **Before fine-tuning**:
  - **Accuracy**: 0.96
  - **Precision**: 0.966
  - **Recall**: 0.946
  - **F1-Score**: 0.954
- **After fine-tuning with `GridSearchCV`**:
  - **Accuracy**: 96.83%
  - **Precision**: 0.98
  - **Recall**: 0.94

### Decision Tree:
- **Accuracy**: 0.89
- **Precision**: 0.890
- **Recall**: 0.873
- **F1-Score**: 0.879

## Conclusion
Based on the results, the **K-Nearest Neighbors (KNN)** model, after fine-tuning, had the highest performance with an **Accuracy** of 96.83% and **Precision** of 0.98. It is the most suitable for this dataset.

The **Logistic Regression** model also provided strong results, with an accuracy of 0.95 and an F1-Score of 0.944. **Naive Bayes** performed reasonably well but slightly underperformed compared to KNN and Logistic Regression. The **Decision Tree** model had the lowest performance with an accuracy of 0.89 and an F1-Score of 0.879.

### Key Insights:
- **KNN** (after fine-tuning) outperformed other models in accuracy and precision, making it the most suitable for this dataset.
- **Logistic Regression** is a strong alternative with good overall performance.
- **Naive Bayes** is a good option but slightly underperforms compared to KNN and Logistic Regression.
- **Decision Tree** had the lowest accuracy and F1-score, indicating it might not be as reliable for this dataset.

These results suggest that **KNN** (after fine-tuning) and **Logistic Regression** are the most effective models for detecting breast cancer in this dataset.
