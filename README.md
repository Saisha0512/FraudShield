# FraudShield : Credit Card Fraud Detection 

## Overview of the Project
**FraudShield** is a Machine Learning project designed to detect **fraudulent credit card transactions**.  
The dataset is highly imbalanced — with only **0.17%** of transactions being fraudulent — making this a **real-world challenge** for classification algorithms.  

The project employs **data balancing using SMOTE (Synthetic Minority Over-sampling Technique)** and compares multiple machine learning models to determine which performs best in detecting fraudulent activities. 

## Tech Stacks & Libraries Used 
- **Language :** Python
- **Libraries :** NumPy, Pandas, Matplotlib, Seaborn, Scikit - learn, Imbalanced - learn, XBGoost

## Dataset
**Source :** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) 

The dataset contains anonymized features (V1–V28) obtained from PCA transformations, along with transaction **Time**, **Amount** and **Class** (target variable) : 
- `Class = 0` → Non-Fraudulent Transaction  
- `Class = 1` → Fraudulent Transaction

**Dataset Statistics :**
- Total Transactions → 284,807  
- Fraudulent Transactions → 492 (~0.17%)  
- Non-Fraudulent Transactions → 284,315 (~99.83%)

## Data Cleaning, Preprocessing & Visualization
1. **Loaded & cleaned** the dataset (no missing values).
2. Plotted a **bar chart to show class imbalance** and a **Seaborn heatmap** to visualize **feature correlations**.
3. **Scaled** `Time` and `Amount` columns using `StandardScaler` for even distribution.  
4. **Split** the data into training and testing sets (80–20 split).  
5. **Balanced** the dataset using **SMOTE** to oversample the minority class (fraud).

## Balancing the data using SMOTE 
**SMOTE (Synthetic Minority Over-sampling Technique)** creates synthetic samples of the minority class instead of duplicating them.  
This helps the model learn meaningful patterns and reduces bias toward the majority class.  

**Before SMOTE:**  
- Non-Fraud: 227,454  
- Fraud: 391  

**After SMOTE:**  
- Non-Fraud: 227,454  
- Fraud: 227,454

This balancing improved model recall and F1 score significantly.

## Machine Learning Models Used 
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**

## Evaluation Metrics for each Model
Each model was evaluated using the following metrics:  
- **Accuracy Score**  
- **Precision**  
- **Recall**  
- **F1 Score**

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|---------|-----------|
| **Logistic Regression** | 0.9747 | 0.0621 | 0.9406 | 0.1164 |
| **Decision Tree** | 0.9977 | 0.4233 | 0.7921 | 0.5517 |
| **Random Forest** | 0.9972 | 0.3750 | 0.8614 | 0.5225 |
| **XGBoost** | 0.9993 | 0.7870 | 0.8416 | 0.8134 |

**Best Performing Models :** **Random Forest** and **XGBoost** achieved the most balanced trade-off between precision and recall — essential for fraud detection.

## Result Visualizations
**Evaluation Metrics :**
<img width="1189" height="989" alt="image" src="https://github.com/user-attachments/assets/fe27b28b-a304-475f-bfa9-708704ca4bd6" />

**Confusion Matrices :**
<img width="1133" height="989" alt="image" src="https://github.com/user-attachments/assets/2518ef94-1296-47b1-a45c-4342ff3db09e" />

## Conclusion 
**FraudShield** demonstrates how Machine Learning can be applied to detect financial fraud effectively.  
By combining **data balancing (SMOTE)**, **ensemble models (RF/XGBoost)** and **strong evaluation metrics**, the project achieves high accuracy and robustness against class imbalance.  

This project showcases the **end-to-end ML pipeline** — from preprocessing and resampling to model building, comparison and visualization.
