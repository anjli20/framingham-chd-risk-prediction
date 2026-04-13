# Framingham CHD Risk Prediction

Predicts 10-year coronary heart disease (CHD) risk using the 
Framingham Heart Study dataset. Built as a complete machine 
learning pipeline from raw data to evaluated, saved models.

## What it does
- Clinical feature engineering on 16 raw patient variables
- Handles class imbalance with SMOTE-ENN resampling
- Trains and cross-validates 7 models: Logistic Regression, 
  Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN
- Builds a stacking ensemble from the top 3 performers
- Tunes the best model via GridSearchCV
- Calibrates probabilities using isotonic regression
- Selects optimal decision threshold (Youden-J and F1-max)
- Explains predictions with SHAP (summary, bar, waterfall plots)
- Tracks all experiments with MLflow
- Benchmarks ML model against the clinical Framingham Risk Score
  using DeLong's statistical AUC test

## Dataset
[Framingham Heart Study](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)  
4,240 patients · 16 features · Binary target (10-year CHD risk)

## Requirements
```bash
pip install imbalanced-learn xgboost lightgbm catboost shap mlflow scipy scikit-learn
```

## Usage
Open in Google Colab or any Jupyter environment and run all cells.  
Models are saved as `.pkl` files on completion.
