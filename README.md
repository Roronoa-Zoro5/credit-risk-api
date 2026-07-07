# Give Me Some Credit – Credit Default Prediction

Binary classification model to predict the probability that a customer will default on their credit within the next two years, using the **Give Me Some Credit** dataset from Kaggle. 
( The data may change as I will make some changes later if needed )

## 1. Problem & Dataset

- Goal: Predict `SeriousDlqin2yrs` (1 = defaulter, 0 = non-defaulter) for each customer.
- Dataset: "Give Me Some Credit" competition data from Kaggle.
- Link: https://www.kaggle.com/c/GiveMeSomeCredit

> Note: Raw data is not included in this repo. Please download it from Kaggle and place the CSV files inside the `data/` folder.

## 2. Approach

Main steps in this project:

1. Data loading and cleaning  
   - Handle missing values (e.g., `MonthlyIncome`, `NumberOfDependents`).  
   - Treat outliers using simple capping/clip strategies.
2. Train–test split  
   - Split into train and test sets (e.g., 80/20), preserving class distribution.
3. Feature scaling  
   - Apply `StandardScaler` (or similar) on numeric features.  
   - Save scaled train/test arrays as CSVs for reuse.
4. Modeling  
   - Baseline model: Logistic Regression.  
   - Evaluation on test set using AUC-ROC, confusion matrix, and classification report.

All steps are implemented in the Jupyter notebooks inside the `notebooks/` directory.

## 3. Results (Baseline)

On the held-out test set (30,000 samples):

- AUC-ROC: **0.79**
- Accuracy: **0.77**
- Class-wise metrics (from `classification_report`):

  - Class 0 (non-default):
    - Precision: ~0.97  
    - Recall: ~0.78  
    - F1-score: ~0.87  

  - Class 1 (default):
    - Precision: ~0.17  
    - Recall: ~0.65  
    - F1-score: ~0.27  

Interpretation:

- The model ranks risky vs safe customers reasonably well (AUC ≈ 0.79) but, at the default 0.5 threshold, it focuses on catching more defaulters (higher recall) at the cost of many false positives.
- This is a **baseline** model; future work can improve both precision and recall for defaulters.

## 4. How to run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/give-me-some-credit-ml.git
cd give-me-some-credit-ml

